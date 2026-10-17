#pragma once

#include "util/pinned.hpp"

#include <atomic>
#include <filesystem>
#include <format>
#include <random>
#include <string>
#include <string_view>
#include <system_error>

// A path that removes itself, so a failing ASSERT leaves nothing behind.  The
// name carries a per-process token: a fixed path in the system temp directory
// is shared mutable state between processes, however private it looks, and two
// copies of one binary would wipe each other's entries through remove_all.
class Scratch : private ddx::impl::pinned {
public:
  explicit Scratch(std::string_view name)
      : path_(std::filesystem::temp_directory_path() /
              std::format("ddx_{}_{:08x}_{}", name, token(), seq())) {
    wipe();
  }
  ~Scratch() { wipe(); }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }
  operator const std::filesystem::path &() const noexcept { // NOLINT
    return path_;
  }
  [[nodiscard]] std::string string() const { return path_.string(); }

private:
  [[nodiscard]] static unsigned token() {
    static const unsigned t = std::random_device{}();
    return t;
  }
  [[nodiscard]] static unsigned seq() {
    static std::atomic<unsigned> s{0};
    return s.fetch_add(1);
  }
  void wipe() const {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  std::filesystem::path path_;
};
