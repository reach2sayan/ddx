#pragma once

#include "util/error.hpp"

#include <format>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ddx::py {

// The bindings use exceptions, unlike libddx, because pybind11 needs them. No
// throw crosses into libddx, which never calls back into Python.
struct PyError : error, std::runtime_error {
  explicit PyError(error e)
      : error(e), std::runtime_error(std::format("{}", e)) {}
  explicit PyError(errc c) : PyError(error{.code = c}) {}
};

[[noreturn]] inline void fail_with(errc c) { throw PyError{c}; }

inline void unwrap(const result<void> &r) {
  if (!r) {
    throw PyError{r.error()};
  }
}

// Excludes void, or a result<void> prvalue would bind here instead of the
// overload above.
template <typename T>
  requires(!std::is_void_v<T>)
[[nodiscard]] T unwrap(result<T> &&r) {
  if (!r) {
    throw PyError{r.error()};
  }
  return std::move(*r);
}

} // namespace ddx::py
