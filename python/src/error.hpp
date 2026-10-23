#pragma once

#include "util/error.hpp"

#include <format>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ddx::py {

// libddx is compiled -fno-exceptions and every refusal travels as a result<T>.
// This TU cannot be, for two reasons: py::error_already_set is itself a C++
// exception, and equation() runs a caller's model under scoped_arena, whose
// guard only the unwind runs.  No throw crosses a -fno-exceptions frame -- the
// callback and the guard are both here, and libddx never calls into Python.
//
// Inherits ddx::error rather than restating it: the code *is* that type and the
// message is what its own formatter says.
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

// Constrained away from void: a result<void> prvalue would bind here in
// preference to the overload above.
template <typename T>
  requires(!std::is_void_v<T>)
[[nodiscard]] T unwrap(result<T> &&r) {
  if (!r) {
    throw PyError{r.error()};
  }
  return std::move(*r);
}

} // namespace ddx::py
