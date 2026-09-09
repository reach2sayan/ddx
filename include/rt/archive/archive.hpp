#pragma once

#include "rt/archive/codec.hpp"
#include "rt/archive/container.hpp"
#include "rt/archive/digest.hpp"
#include "rt/archive/snapshot.hpp"
#include "rt/archive/sound.hpp"
#include "rt/opcode.hpp"
#include "util/error.hpp"
#include "util/export.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// A built equation on disk.  Three entry points and nothing else: save, load
// and verify are objects, so none of them can be overloaded, specialised, or
// found by argument-dependent lookup on a type that happens to live nearby.
namespace ddx::rt {

#ifdef DDX_HAS_JIT
// One lane's kernel as a file carries it.
template <impl::Numeric T>
[[nodiscard]] auto
object_of(Want want, const Graph<T> &graph, const jit::Kernel &kernel,
          const jit::Compiler &compiler, const jit::Codegen &codegen) {
  const auto code = kernel.object();
  return Object{.want = want,
                .symbol = std::string{kernel.symbol()},
                .host = std::string{compiler.host_identity()},
                .digest = digest(graph),
                .codegen = codegen,
                .code = {code.begin(), code.end()}};
}

// What a file gave this lane back.  A miss and a refused link are the same
// answer: an empty kernel, and a compile still to do.
struct Adopted {
  jit::Kernel kernel;
  jit::Level level = jit::Level::O0;
  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(kernel);
  }
};

// The shape is taken from the graph just frozen, never from the file: a forged
// entry may supply code and a symbol but cannot claim a column count.
template <impl::Numeric T>
[[nodiscard]] Adopted
adopt_stored(std::span<const Object> objects, Want want, const Graph<T> &graph,
             jit::Compiler &compiler, const jit::Codegen &codegen) {
  const Object *const have = find_object(objects, want, digest(graph),
                                         compiler.host_identity(), codegen);
  if (have == nullptr) {
    return {};
  }

  if (auto adopted =
          compiler.adopt(have->code, have->symbol, jit::KernelShape::of(graph));
      !adopted) {
    return {};
  } else {
    return Adopted{.kernel = std::move(*adopted),
                   .level = have->codegen.codegen_level};
  }
}
#endif

namespace detail {

struct SaveFn;
template <std::floating_point T> struct LoadSnapshotFn;
template <std::floating_point T> struct VerifyFn;

// What the three entry points are built from, and what nothing else may call.
class Format {
  friend struct SaveFn;
  template <std::floating_point> friend struct LoadSnapshotFn;
  template <std::floating_point> friend struct VerifyFn;

  static constexpr std::string_view magic = "ddxgraph";
  // Bumped by hand only for what neither stamp can see: a reordered section, a
  // change in coverage.  A new prologue or payload field no longer needs it --
  // Container::stamp() folds both shapes and refuses the old file itself.
  static constexpr std::uint32_t format = 3;

  // The prologue this build would write, against the one the file carries.
  template <std::floating_point T>
  [[nodiscard]] static result<void> compatible(const FileHeader &h) {
    if (h.format != format || h.schema != Container::stamp<Snapshot<T>>() ||
        h.scalar_size != sizeof(T) || h.scalar_kind != ScalarKind::Floating) {
      return fail(errc::bad_archive);
    }
    return {};
  }

  // The file's opcode byte to this build's, or Reader::no_op where the file
  // names one this build does not have.
  [[nodiscard]] static std::vector<std::uint8_t>
  remap_of(std::span<const std::string> labels) {
    std::vector<std::uint8_t> remap(labels.size(), Reader::no_op);
    for (const auto [i, label] : labels | std::views::enumerate) {
      if (const auto op = opcode_of(label)) {
        remap[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(*op);
      }
    }
    return remap;
  }
};

struct SaveFn {
  template <std::floating_point T>
  [[nodiscard]] result<void>
  operator()(const Snapshot<T> &snap, const std::filesystem::path &path) const {
    // Together, and in this order: the table is what every opcode byte *means*,
    // so outside the checksum one flipped bit reinterprets the graph, and it
    // has to be readable before a graph byte is.
    const auto body = Container::encode(opcode_labels(), snap);

    const FileHeader h{
        .magic = Format::magic,
        .format = Format::format,
        .schema = Container::stamp<Snapshot<T>>(),
        .scalar_size = sizeof(T),
        .scalar_kind = ScalarKind::Floating,
        .model_nodes = snap.model_nodes,
        .model_digest = digest<T>(snap.symbols, snap.nodes, snap.model_nodes)};
    return Container::write(path, Container::pack(Format::magic, h, body));
  }
};

template <std::floating_point T> struct LoadSnapshotFn {
  [[nodiscard]] result<Verified<T>>
  operator()(const std::filesystem::path &path) const {
    return Container::read(path).and_then(opened).and_then(trusted);
  }

private:
  // The prologue and the payload span it introduces, still borrowing the file's
  // bytes.
  using Bytes = std::pair<FileHeader, std::span<const std::byte>>;

  // What survives the file: the prologue it came with, and the graph it
  // carried.  Neither borrows the bytes any more, so nothing downstream has a
  // lifetime to keep.
  struct Loaded {
    FileHeader head;
    Snapshot<T> snap;
  };

  // Bytes to a graph.  The buffer is this function's parameter, so it outlives
  // every span taken from it and no caller above ever holds one -- which is the
  // whole reason this step exists rather than being three more links in the
  // chain.  unpack clears the checksum first: nothing here is parsed until it
  // does, corruption in bytes that name things changing what the rest of them
  // mean rather than breaking them.
  [[nodiscard]] static result<Loaded> opened(std::vector<std::byte> bytes) {
    return Container::unpack(bytes, Format::magic)
        .and_then(supported)
        .and_then(decoded);
  }

  // The prologue this build would have written, against the one the file
  // carries.
  [[nodiscard]] static result<Bytes> supported(Bytes found) {
    return Format::compatible<T>(found.first).transform([&found] {
      return found;
    });
  }

  // The label table, then its inverse against this build's enum, then the graph
  // read through it.  The table carries its own length, so the prologue does
  // not repeat it, and it is read first because an opcode byte means nothing
  // until it is.
  [[nodiscard]] static result<Loaded> decoded(Bytes found) {
    const auto body = found.second;
    Reader tr{body, {}};
    std::vector<std::string> labels;
    wire(tr, labels);
    if (!tr.ok()) {
      return fail(errc::archive_corrupt);
    }
    // Named, not the call: Reader spans the remap rather than owning it, and a
    // temporary would be gone before the first opcode byte is read through it.
    const auto remap = Format::remap_of(labels);
    Loaded out{.head = found.first, .snap = {}};
    Reader r{body.last(tr.remaining()), remap};
    wire(r, out.snap);
    // Whole or not at all: bytes left over are as much a corrupt file as a
    // reader running short.
    if (!r.ok() || !r.spent()) {
      return fail(errc::archive_corrupt);
    }
    return out;
  }

  // Only now is any of it believed: the prologue fields no checksum could
  // cover, then the invariants nothing downstream re-tests -- which is what
  // makes it a Verified.
  [[nodiscard]] static result<Verified<T>> trusted(Loaded l) {
    return keyed(l.head, l.snap).and_then([&l] {
      return verified(std::move(l.snap));
    });
  }

  // Every prologue field is verified against what the payload says, and none is
  // written that is not.  `model_nodes` is duplicated under the checksum and
  // would otherwise be written, never read, and one edit from being trusted.
  [[nodiscard]] static result<void> keyed(const FileHeader &head,
                                          const Snapshot<T> &snap) {
    const bool agrees = head.model_nodes == snap.model_nodes &&
                        snap.model_nodes <= snap.nodes.size() &&
                        digest<T>(snap.symbols, snap.nodes, snap.model_nodes) ==
                            head.model_digest;
    return agrees ? result<void>{} : fail(errc::archive_corrupt);
  }
};

// Whether this build can read the file at all: prologue, checksum, structural
// invariants.  *Which* equation it holds is Equation::verify().
template <std::floating_point T> struct VerifyFn {
  [[nodiscard]] result<void> operator()(const std::filesystem::path &p) const {
    return LoadSnapshotFn<T>{}(p).transform([](const Verified<T> &) {});
  }
};

} // namespace detail

// The whole public surface.  Objects rather than function templates, so the
// scalar rides on the object: `load_snapshot<float>(p)`, and `load_snapshot<>`
// where the default will do.  `save` deduces T from its snapshot.
inline constexpr detail::SaveFn save{};
template <std::floating_point T = double>
inline constexpr detail::LoadSnapshotFn<T> load_snapshot{};
template <std::floating_point T = double>
inline constexpr detail::VerifyFn<T> verify{};

} // namespace ddx::rt
