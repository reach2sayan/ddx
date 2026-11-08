#pragma once

#include "ops/operations.hpp" // pow_impl, atan2_impl, hypot_impl, max_impl, min_impl, abs_impl
#include "ops/unary_math.hpp" // the unary math descriptors
#include "rt/opcode.hpp"
#include "util/config.hpp"
#include "util/meta.hpp"

#include <concepts> // std::equality_comparable, std::same_as
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace ddx::rt {

namespace detail {

template <typename Fn>
inline constexpr bool is_field_op_v = meta::contains(
    meta::list_c<std::plus<>, std::multiplies<>, std::divides<>, std::negate<>>,
    meta::type_c<Fn>);

#define DDX_RT_PROBE1(fn, Op, label, ...)                                      \
  template <impl::Numeric T>                                                   \
  inline constexpr bool probes_##Op =                                          \
      impl::CArithmetic<T> || requires(const T &u) { fn(u); };
DDX_UNARY_MATH_TABLE(DDX_RT_PROBE1)
#undef DDX_RT_PROBE1

// Probes what each binary functor's body calls, not the row's spelling:
// extremum_impl never calls max(a, b), it compares and takes a midpoint.
template <typename Fn, impl::Numeric T>
inline constexpr bool binary_probe_v = false;
template <impl::Numeric T>
inline constexpr bool binary_probe_v<impl::detail::pow_impl, T> =
    requires(const T &a, const T &b) { pow(a, b); };
template <impl::Numeric T>
inline constexpr bool binary_probe_v<impl::detail::atan2_impl, T> =
    requires(const T &a, const T &b) { atan2(a, b); };
template <impl::Numeric T>
inline constexpr bool binary_probe_v<impl::detail::hypot_impl, T> =
    requires(const T &a, const T &b) { hypot(a, b); };
template <bool IsMax, impl::Numeric T>
inline constexpr bool binary_probe_v<impl::detail::extremum_impl<IsMax>, T> =
    std::equality_comparable<T> && requires(const T &a, const T &b) {
      { a < b } -> std::convertible_to<bool>;
      midpoint(a, b);
    };

#define DDX_RT_PROBE2(fn, Op, label, functor, ...)                             \
  template <impl::Numeric T>                                                   \
  inline constexpr bool probes_##Op =                                          \
      impl::CArithmetic<T> || binary_probe_v<functor, T>;
DDX_RT_BINARY_TABLE(DDX_RT_PROBE2)
#undef DDX_RT_PROBE2

template <impl::Numeric T>
inline constexpr bool probes_Abs =
    impl::CArithmetic<T> || requires(const T &u) { abs(u); };

template <impl::Numeric T> inline constexpr bool probes_Neg = true;
// Comparisons need only ordering; select needs only a test against zero.
template <impl::Numeric T>
inline constexpr bool probes_Lt =
    impl::CArithmetic<T> || std::totally_ordered<T>;

template <impl::Numeric T> inline constexpr bool probes_Le = probes_Lt<T>;

template <impl::Numeric T>
inline constexpr bool probes_Select =
    impl::CArithmetic<T> || std::equality_comparable<T>;
template <impl::Numeric T>
inline constexpr bool probes_VSelect = probes_Select<T>;

template <impl::Numeric T> inline constexpr bool probes_Sign = probes_Lt<T>;

// Every row instantiates for every T, so an op the scalar lacks returns T{}.
// Callers check supports() before relying on the result.
template <typename Fn, impl::Numeric T, bool Ok>
[[nodiscard]] constexpr T
supported([[maybe_unused]] const impl::Numeric auto &...args) noexcept {
  if constexpr (Ok) {
    return static_cast<T>(Fn{}(args...));
  } else {
    return T{};
  }
}

// One table row as a type: the functor, whether T supports it, and its arity.
// A leaf row has a void functor.
template <typename Fn, bool Ok, std::size_t Arity> struct Row {
  using functor = Fn;
  static constexpr bool ok = Ok;
  static constexpr std::size_t arity = Arity;
};
// Field ops (+, *, /, negate) are supported at every T regardless of probe.
template <typename Fn, bool Probed, std::size_t Arity>
using row_t = Row<Fn, Probed || is_field_op_v<Fn>, Arity>;

// The single switch over the opcode tables. Every per-opcode query is a
// visitor over the Row, so the operation set is spelled out only here.
template <impl::Numeric T>
[[nodiscard]] DDX_ALWAYS_INLINE constexpr decltype(auto)
dispatch(OpCode op, auto &&visit) {
  switch (op) {
#define DDX_RT_DISPATCH(fn, Op, label, functor, ...)                           \
  case OpCode::Op:                                                             \
    return visit(row_t<functor, probes_##Op<T>, arity_of(OpCode::Op)>{});
    DDX_RT_UNARY_TABLE(DDX_RT_DISPATCH)
    DDX_RT_BINARY_TABLE(DDX_RT_DISPATCH)
    DDX_RT_COMPARE_TABLE(DDX_RT_DISPATCH)
    DDX_RT_TERNARY_TABLE(DDX_RT_DISPATCH)
#undef DDX_RT_DISPATCH
#define DDX_RT_DISPATCH(fn, Op, label)                                         \
  case OpCode::Op:                                                             \
    return visit(Row<impl::detail::Op##Fn<T>, probes_##Op<T>, 1>{});
    DDX_UNARY_MATH_TABLE(DDX_RT_DISPATCH)
#undef DDX_RT_DISPATCH
#define DDX_RT_DISPATCH(fn, Op, label, ...)                                    \
  case OpCode::Op:                                                             \
    return visit(Row<void, true, 0>{});
    DDX_RT_LEAF_TABLE(DDX_RT_DISPATCH)
#undef DDX_RT_DISPATCH
    // Extern calls are unsupported at every scalar so Taylor sweeps avoid
    // them. N-ary folds use a field op, so they are supported everywhere.
#define DDX_RT_DISPATCH(fn, Op, label, arity, kind, ...)                       \
  case OpCode::Op:                                                             \
    return visit(Row<void, Kind::kind == Kind::Nary, arity>{});
    DDX_RT_EXTERN_TABLE(DDX_RT_DISPATCH)
    DDX_RT_NARY_TABLE(DDX_RT_DISPATCH)
#undef DDX_RT_DISPATCH
    // Tensor rows run only through the interpreter's kernel table.
#define DDX_RT_DISPATCH(fn, Op, label, arity, ...)                             \
  case OpCode::Op:                                                             \
    return visit(Row<void, false, arity>{});
    DDX_RT_TENSOR_TABLE(DDX_RT_DISPATCH)
#undef DDX_RT_DISPATCH
  }
  std::unreachable();
}

} // namespace detail

// Whether apply<T>(op, ...) computes op or returns T{}. Leaves are always
// supported.
template <impl::Numeric T>
[[nodiscard]] constexpr bool supports(OpCode op) noexcept {
  return detail::dispatch<T>(op, []<typename R>(R) { return R::ok; });
}

// Computes op at double and builds a node at Expression, so the interpreter,
// folding and arena share it. A wrong arity or unsupported op returns T{}.
template <impl::Numeric T>
[[nodiscard]] constexpr T apply(OpCode op,
                                const std::same_as<T> auto &...args) noexcept
  requires(sizeof...(args) >= 1 && sizeof...(args) <= 3)
{
  return detail::dispatch<T>(op, [&]<typename R>(R) -> T {
    if constexpr (R::arity == sizeof...(args) &&
                  !std::is_void_v<typename R::functor>) {
      return detail::supported<typename R::functor, T, R::ok>(args...);
    } else {
      return T{};
    }
  });
}

} // namespace ddx::rt
