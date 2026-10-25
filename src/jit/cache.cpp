#include "cache.hpp"

#include <boost/crc.hpp>
#include <boost/describe.hpp>
#include <boost/endian/conversion.hpp>
#include <boost/hash2/fnv1a.hpp>
#include <boost/hash2/hash_append.hpp>
#include <boost/integer.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/scope/scope_exit.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <concepts>
#include <cstring>
#include <format>
#include <fstream>
#include <random>
#include <ranges>
#include <system_error>
#include <thread>
#include <type_traits>

namespace ddx::jit::detail {
namespace {

using flavor = boost::hash2::little_endian_flavor;

template <typename U>
using described_members =
    boost::describe::describe_members<U, boost::describe::mod_any_access |
                                             boost::describe::mod_inherited>;

// The same bits as an unsigned of the same width: hash2 appends a float as
// v + 0, which maps -0.0 onto +0.0, and a graph holding -0.0 must not load
// +0.0's kernel.  A described type folds member by member, a range its length
// then its elements -- never a memcpy, since NodeRecord has padding.
constexpr void fold(boost::hash2::fnv1a_64 &h, const auto &v) {
  using U = std::remove_cvref_t<decltype(v)>;
  if constexpr (std::same_as<U, std::string>) {
    boost::hash2::hash_append(h, flavor{}, v);
  } else if constexpr (boost::describe::has_describe_members<U>::value) {
    boost::mp11::mp_for_each<described_members<U>>(
        [&](auto D) { fold(h, v.*D.pointer); });
  } else if constexpr (std::ranges::range<U>) {
    boost::hash2::hash_append(h, flavor{},
                              static_cast<std::uint64_t>(std::ranges::size(v)));
    std::ranges::for_each(v, [&h](const auto &e) { fold(h, e); });
  } else {
    boost::hash2::hash_append(
        h, flavor{},
        std::bit_cast<typename boost::uint_t<8 * sizeof(U)>::exact>(v));
  }
}

// Bumped by hand for what neither the digest nor the entry can see: the
// emitter, the pass pipeline, the ABI, the opcode numbering the digest folds.
// Folded into the key, so a bump misses rather than refuses.
constexpr std::uint32_t kEpoch = 5;

// The file: a tag, the format, the key, a CRC over everything after it, the
// two lengths, then the symbol and the object.  Little-endian throughout.
constexpr std::string_view kMagic = "ddxjitob";
constexpr std::uint32_t kFormat = 1;
constexpr std::size_t kHeaderBytes = 8 + 4 + 8 + 4 + 4 + 8;

template <std::unsigned_integral U> void put(std::span<std::byte> at, U v) {
  boost::endian::native_to_little_inplace(v);
  std::memcpy(at.data(), &v, sizeof v);
}
template <std::unsigned_integral U>
[[nodiscard]] U get(std::span<const std::byte> at) {
  U v{};
  std::memcpy(&v, at.data(), sizeof v);
  return boost::endian::little_to_native(v);
}

[[nodiscard]] std::uint32_t checksum(std::span<const std::byte> bytes) {
  boost::crc_32_type crc;
  crc.process_bytes(bytes.data(), bytes.size());
  return crc.checksum();
}

// Per-process and per-call: drawn once so it is not paid per write, counted
// so one process cannot collide with itself.
[[nodiscard]] std::string stage_suffix() {
  static const auto salt = std::random_device{}();
  static std::atomic<std::uint32_t> seq{0};
  return std::format(".{:08x}{:08x}.tmp", salt, seq.fetch_add(1));
}

// Windows will not replace a destination another handle still has open, and
// racing writers are exactly that: the loser gets a sharing violation instead
// of losing an atomic last-writer-wins.  Every hold is brief, so a bounded
// retry restores the property the staging name was chosen for.  POSIX rename
// never fails this way, and there the loop never comes round.
[[nodiscard]] bool rename_over(const std::filesystem::path &from,
                               const std::filesystem::path &to) {
  using namespace std::chrono_literals;
  auto backoff = 1ms;
  for (int attempt = 0; attempt < 20; ++attempt) {
    std::error_code ec;
    std::filesystem::rename(from, to, ec);
    if (!ec) {
      return true;
    }
    if (attempt < 4) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(backoff);
      backoff = std::min(backoff * 2, 32ms);
    }
  }
  return false;
}

[[nodiscard]] std::optional<std::vector<std::byte>>
read_file(const std::filesystem::path &path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec || !std::filesystem::is_regular_file(path, ec)) {
    return std::nullopt;
  }
  std::ifstream in{path, std::ios::binary};
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  in.read(reinterpret_cast<char *>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  if (!in || in.gcount() != static_cast<std::streamsize>(bytes.size())) {
    return std::nullopt;
  }
  return bytes;
}

// Staged beside the target and renamed over it: rename is atomic only within
// one filesystem, and the suffix is unique per writer so two processes never
// truncate each other's partial write.  Last writer wins; a reader sees one
// whole file.
void write_file(const std::filesystem::path &path,
                std::span<const std::byte> bytes) {
  std::filesystem::path tmp = path;
  tmp += stage_suffix();
  boost::scope::scope_exit unstage{[&tmp] {
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
  }};
  {
    std::ofstream out{tmp, std::ios::binary | std::ios::trunc};
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.flush();
    if (!out) {
      return;
    }
  }
  if (rename_over(tmp, path)) {
    unstage.set_active(false);
  }
}

} // namespace

template <CKernelScalar T> std::uint64_t digest(const rt::Graph<T> &g) {
  boost::hash2::fnv1a_64 h;
  // The element type, which no node's bits can stand for: 1.0 in a float
  // graph and in a double graph emit different kernels.
  fold(h, std::uint8_t{sizeof(T)});
  const rt::Inputs &in = g.inputs();
  fold(h, in.by_kind);
  fold(h, in.directions);
  fold(h, g.layout());
  const auto schedule = g.schedule();
  fold(h, schedule.size());
  // A step's contraction is the graph's, decided at freeze: the same nodes
  // emit fma under one and a multiply and an add under the other.
  for (const rt::Step &s : schedule) {
    fold(h, s.node);
    fold(h, g[s.node]);
    std::ranges::for_each(g.operands(s.node),
                          [&h](rt::NodeId u) { fold(h, u); });
    fold(h, std::array{s.fma.x, s.fma.y, s.fma.z});
    fold(h, s.fma.negated);
    fold(h, s.fused);
  }
  std::ranges::for_each(g.outputs(), [&h](rt::NodeId o) { fold(h, o); });
  fold(h, g.lists()); // a has_list node's operands live here, not in operands()
  fold(h, g.externs()); // arity and order size the call's answer
  return h.result();
}

template std::uint64_t digest(const rt::Graph<double> &);
template std::uint64_t digest(const rt::Graph<float> &);

// The width *emitted* in place of the one asked for: a derived `Lanes` means
// the host's, so the raw request would give one graph two keys.
std::uint64_t cache_key(std::uint64_t graph, std::string_view host,
                        const Codegen &emitted) {
  boost::hash2::fnv1a_64 h;
  boost::hash2::hash_append(h, flavor{}, host);
  fold(h, graph);
  fold(h, kEpoch);
  fold(h, emitted);
  return h.result();
}

std::filesystem::path entry_path(std::string_view dir, std::uint64_t key) {
  return std::filesystem::path{dir} / std::format("{:016x}.ddxjit", key);
}

// Every prologue field is checked by exact compare -- the prologue is the one
// part no checksum covers -- and the lengths must account for every byte left,
// so a forged length is a miss before anything is sliced by it.
std::optional<Entry> read_entry(std::string_view dir, std::uint64_t key) {
  const auto bytes = read_file(entry_path(dir, key));
  if (!bytes || bytes->size() < kHeaderBytes) {
    return std::nullopt;
  }
  const std::span<const std::byte> all{*bytes};
  if (std::memcmp(all.data(), kMagic.data(), kMagic.size()) != 0 ||
      get<std::uint32_t>(all.subspan(8)) != kFormat ||
      get<std::uint64_t>(all.subspan(12)) != key ||
      get<std::uint32_t>(all.subspan(20)) != checksum(all.subspan(24))) {
    return std::nullopt;
  }
  const auto symbol_bytes = std::size_t{get<std::uint32_t>(all.subspan(24))};
  const auto code_bytes = get<std::uint64_t>(all.subspan(28));
  const auto rest = all.subspan(kHeaderBytes);
  if (symbol_bytes == 0 || symbol_bytes > rest.size() ||
      code_bytes != rest.size() - symbol_bytes || code_bytes == 0) {
    return std::nullopt;
  }
  const auto symbol = rest.first(symbol_bytes);
  const auto code = rest.subspan(symbol_bytes);
  return Entry{
      .symbol = {reinterpret_cast<const char *>(symbol.data()), symbol.size()},
      .code = {code.begin(), code.end()}};
}

void write_entry(std::string_view dir, std::uint64_t key,
                 std::string_view symbol, std::span<const std::byte> code) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    return; // A cache that cannot be written is a cache that always misses.
  }
  std::vector<std::byte> file(kHeaderBytes + symbol.size() + code.size());
  const std::span<std::byte> all{file};
  std::memcpy(all.data(), kMagic.data(), kMagic.size());
  put(all.subspan(8), kFormat);
  put(all.subspan(12), key);
  put(all.subspan(24), static_cast<std::uint32_t>(symbol.size()));
  put(all.subspan(28), static_cast<std::uint64_t>(code.size()));
  std::ranges::copy(std::as_bytes(std::span{symbol}),
                    all.subspan(kHeaderBytes).begin());
  std::ranges::copy(code, all.subspan(kHeaderBytes + symbol.size()).begin());
  put(all.subspan(20), checksum(all.subspan(24)));
  write_file(entry_path(dir, key), file);
}

} // namespace ddx::jit::detail
