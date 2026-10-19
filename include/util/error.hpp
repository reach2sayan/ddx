#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <ostream>
#include <string_view>
#include <utility>

namespace ddx {

// clang-format off
#define DDX_ERRC_TABLE(X)                                                                     \
  /* A point that does not match the expression's symbols. */                                 \
  X(short_point,        "the point supplies fewer values than the expression has inputs")     \
  X(wrong_arity,        "the point supplies one value per input, and this one does not")      \
  X(wrong_direction,    "the direction supplies one value per symbol, and this one does not") \
  X(unknown_symbol,     "no symbol of that name")                                             \
  X(index_out_of_range, "the index does not name a symbol of this expression")                \
  X(wrong_column_count, "wrong number of output columns")                                     \
                                                                                              \
  /* Runtime graphs. */                                                                       \
  X(outside_equation, "no equation is being built -- name symbols inside ddx::rt::equation()")\
  X(no_graph,         "the expression has no graph (a bare literal names none)")              \
  X(name_clash,       "the name is already an input of another kind")                          \
  X(closed_twice,     "an implicit input is named, with a residual, twice")                   \
  X(nested_implicit,  "a residual reaches an implicit input other than its own")              \
  X(extern_order,     "an extern function does not supply the partials this call needs")       \
  X(extern_shape,     "an extern function's arity or order cannot be represented")            \
  X(not_univariate,   "needs exactly one symbol")                                             \
  X(not_single_output, "needs exactly one output, and this is a system")                      \
  X(unsupported_scalar, "an operation the sweep's scalar type does not define")               \
  X(tensor_shape,     "a tensor's shape does not fit the operation, or one sits where a scalar must") \
  X(tensor_kernel,    "a graph holding tensor nodes runs on the sweep; no kernel is built for it") \
                                                                                              \
  /* The JIT. */                                                                              \
  X(jit_target, "the JIT could not bring up a target for this host")                          \
  X(jit_module, "the JIT could not take the emitted module")                                  \
  X(jit_object, "the JIT could not link an object compiled earlier")                          \
  X(jit_verify, "the emitted module failed verification")                                     \
  X(jit_lookup, "the compiled kernel has no such symbol")                                     \
                                                                                              \
  /* The text surface. */                                                                     \
  X(bad_syntax,       "not an expression this grammar accepts")                               \
  X(unknown_function, "no function of that name")                                             \
  X(wrong_argument_count, "the function takes a different number of arguments")
// clang-format on

enum class errc : std::uint8_t {
#define DDX_ERRC_ENUMERATOR(name, text) name,
  DDX_ERRC_TABLE(DDX_ERRC_ENUMERATOR)
#undef DDX_ERRC_ENUMERATOR
};

// No message string and no source location: an error travels the numeric path,
// where an allocation is as unwelcome as the throw it replaces.  The text sits
// in a static table the formatter reads.
struct error {
  errc code;
  [[nodiscard]] friend constexpr bool operator==(error,
                                                 error) noexcept = default;
};

namespace detail {

inline constexpr std::array kMessages{
#define DDX_ERRC_MESSAGE(name, text) std::string_view{text},
    DDX_ERRC_TABLE(DDX_ERRC_MESSAGE)
#undef DDX_ERRC_MESSAGE
};

[[nodiscard]] constexpr std::string_view message(const errc c) noexcept {
  const auto i = static_cast<std::size_t>(c);
  return i < kMessages.size() ? kMessages[i] : "?";
}
} // namespace detail

inline std::ostream &operator<<(std::ostream &out, const errc c) {
  return out << detail::message(c);
}

inline std::ostream &operator<<(std::ostream &out, const error e) {
  return out << e.code;
}

template <typename T> using result = std::expected<T, error>;
[[nodiscard]] constexpr std::unexpected<error> fail(const errc c) noexcept {
  return std::unexpected{error{.code = c}};
}

} // namespace ddx

// Deriving from the string_view formatter, so a caller's "{:>16}" reaches the
// text.
template <>
struct std::formatter<ddx::errc, char>
    : std::formatter<std::string_view, char> {
  auto format(const ddx::errc c, std::format_context &ctx) const {
    return std::formatter<std::string_view, char>::format(
        ddx::detail::message(c), ctx);
  }
};

template <>
struct std::formatter<ddx::error, char> : std::formatter<ddx::errc, char> {
  auto format(const ddx::error e, std::format_context &ctx) const {
    return std::formatter<ddx::errc, char>::format(e.code, ctx);
  }
};
