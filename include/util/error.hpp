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
  X(short_point,        "the point supplies fewer values than the expression has symbols")    \
  X(wrong_arity,        "the point supplies one value per symbol, and this one does not")     \
  X(wrong_direction,    "the direction supplies one value per symbol, and this one does not") \
  X(unknown_symbol,     "no symbol of that name")                                             \
  X(index_out_of_range, "the index does not name a symbol of this expression")                \
  X(wrong_column_count, "wrong number of output columns")                                     \
                                                                                              \
  /* Runtime graphs. */                                                                       \
  X(no_arena,         "no arena is current -- name symbols inside ddx::rt::equation()")       \
  X(no_graph,         "the expression has no graph (a bare literal names none)")              \
  X(sealed_arena,     "the arena already backs an equation, so its symbols are final")        \
  X(not_univariate,   "needs exactly one symbol")                                             \
  X(unsupported_scalar, "an operation the sweep's scalar type does not define")               \
                                                                                              \
  /* Saved graphs. */                                                                         \
  X(archive_io,       "the file could not be read or written")                                \
  X(bad_archive,      "not a ddx graph file, or a format this build does not read")           \
  X(archive_corrupt,  "the file's checksum or its structure does not hold")                   \
  X(archive_mismatch, "the file loads, but does not describe this equation")                  \
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
  X(wrong_argument_count, "the function takes a different number of arguments")          \
                                                                                              \
  /* The device. */                                                                           \
  X(no_device,      "no OpenCL device with double precision answers the selector")            \
  X(device_compile, "the device's compiler refused the emitted kernel")                       \
  X(device_launch,  "the device could not run the kernel or move its columns")
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
