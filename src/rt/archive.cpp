// The part of the archive carrying no scalar: prologue, checksum, file and
// opcode label table.  Out of line like coupling.cpp and dot.cpp, or it would
// be emitted once per scalar for byte-identical code.
#include "rt/archive/archive.hpp"

#include <boost/crc.hpp>
#include <boost/scope/scope_exit.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <format>
#include <fstream>
#include <random>
#include <ranges>
#include <system_error>
#include <thread>
#include <utility>

namespace ddx::rt::detail {
namespace {

// Per-process and per-call, in standard C++ rather than getpid(): drawn once so
// it is not paid per write, counted so one process cannot collide with itself.
[[nodiscard]] std::string stage_suffix() {
  static const auto salt = std::random_device{}();
  static std::atomic<std::uint32_t> seq{0};
  return std::format(".{:08x}{:08x}.tmp", salt, seq.fetch_add(1));
}

// Windows will not replace a destination that another handle still has open,
// and racing writers are exactly that: MoveFileEx opens the target for delete,
// so the loser gets a sharing violation instead of losing an atomic
// last-writer-wins.  Every hold is brief -- another rename in flight, or a
// scanner on a freshly written file -- so a bounded retry restores the property
// the staging name was chosen for.  POSIX rename never fails this way, and
// there the loop never comes round.
//
// Retried whatever the error says, rather than on the sharing violation alone:
// the staged file is written and closed and sits in the target's own directory,
// so the remaining ways to fail here are exotic, and one that cannot recover
// only spends the budget before failing exactly as it would have.
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
    // Yielding first: the common contender is a rename mid-flight, and
    // sleeping a whole tick for it would cost more than the write did.
    if (attempt < 4) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(backoff);
      backoff = std::min(backoff * 2, 32ms);
    }
  }
  return false;
}

// The prologue's described fields, between the tag and the reserved tail.
inline constexpr std::size_t field_bytes = Container::header_bytes -
                                           Container::magic_bytes -
                                           Container::reserved_bytes;

} // namespace

void Container::put_header(const FileHeader &h, std::span<std::byte> into) {
  assert(into.size() >= Container::header_bytes);
  std::ranges::fill(into.first(Container::header_bytes),
                    std::byte{}); // the tail included
  std::memcpy(into.data(), h.magic.data(),
              std::min(h.magic.size(), Container::magic_bytes));
  // The same Writer the payload uses, over the same described list get_header
  // reads: there is no second field sequence to keep in step with this one.
  std::vector<std::byte> fields;
  fields.reserve(field_bytes);
  Writer w{fields};
  wire(w, std::as_const(h));
  assert(fields.size() == field_bytes);
  std::ranges::copy(fields, into.begin() + Container::magic_bytes);
}

result<FileHeader> Container::get_header(std::span<const std::byte> bytes,
                                         std::string_view magic) {
  if (bytes.size() < Container::header_bytes) {
    return fail(errc::bad_archive);
  }
  if (std::memcmp(bytes.data(), magic.data(),
                  std::min(magic.size(), Container::magic_bytes)) != 0) {
    return fail(errc::bad_archive);
  }
  // Letting a nonzero reserved byte past would make claiming it later a silent
  // format change.
  if (!std::ranges::all_of(bytes.subspan(Container::magic_bytes + field_bytes,
                                         Container::reserved_bytes),
                           [](std::byte b) { return b == std::byte{}; })) {
    return fail(errc::bad_archive);
  }

  FileHeader h{.magic = magic};
  Reader r{bytes.subspan(Container::magic_bytes, field_bytes), {}};
  wire(r, h);
  if (!r.ok() || !r.spent()) {
    return fail(errc::bad_archive);
  }
  return h;
}

std::uint32_t Container::checksum(std::span<const std::byte> bytes) {
  boost::crc_32_type crc;
  crc.process_bytes(bytes.data(), bytes.size());
  return crc.checksum();
}

result<std::vector<std::byte>>
Container::read(const std::filesystem::path &path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec || !std::filesystem::is_regular_file(path, ec)) {
    return fail(errc::archive_io);
  }
  std::ifstream in{path, std::ios::binary};
  if (!in) {
    return fail(errc::archive_io);
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  in.read(reinterpret_cast<char *>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
    return fail(errc::archive_io);
  }
  return bytes;
}

result<void> Container::write(const std::filesystem::path &path,
                              std::span<const std::byte> bytes) {
  // Beside the target: rename is only atomic within a filesystem, and one
  // across a mount boundary would silently become a copy.  The suffix is unique
  // per writer -- with a shared ".tmp" two processes truncate each other's
  // partial write and both rename it.  Distinct names make the loser's rename a
  // no-op: last writer wins, and every reader sees one whole file.
  std::filesystem::path tmp = path;
  tmp += stage_suffix();
  // Until the rename lands, every way out removes what was staged.
  boost::scope::scope_exit unstage{[&tmp] {
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
  }};
  {
    std::ofstream out{tmp, std::ios::binary | std::ios::trunc};
    if (!out) {
      return fail(errc::archive_io);
    }
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.flush();
    if (!out) {
      return fail(errc::archive_io);
    }
  }
  if (!rename_over(tmp, path)) {
    return fail(errc::archive_io);
  }
  unstage.set_active(false);
  return {};
}

std::vector<std::byte> Container::pack(std::string_view magic,
                                       const FileHeader &h,
                                       std::span<const std::byte> payload) {
  FileHeader stamped = h;
  stamped.magic = magic;
  stamped.payload_crc = Container::checksum(payload);
  std::vector<std::byte> file;
  file.reserve(Container::header_bytes + payload.size());
  file.resize(Container::header_bytes);
  Container::put_header(stamped, file);
  file.insert(file.end(), payload.begin(), payload.end());
  return file;
}

result<std::pair<FileHeader, std::span<const std::byte>>>
Container::unpack(std::span<const std::byte> bytes, std::string_view magic) {
  const auto h = Container::get_header(bytes, magic);
  if (!h) {
    return std::unexpected{h.error()};
  }

  if (const auto payload = bytes.subspan(Container::header_bytes);
      Container::checksum(payload) != h->payload_crc) {
    return fail(errc::archive_corrupt);
  } else {
    return std::pair{*h, payload};
  }
}

} // namespace ddx::rt::detail
