#pragma once

#include "ops/operations.hpp" // pow_impl, atan2_impl, hypot_impl, max_impl, min_impl, abs_impl
#include "ops/unary_math.hpp" // the eighteen descriptors
#include "rt/opcode.hpp"
#include "util/config.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <concepts> // std::equality_comparable, std::same_as
#include <cstddef>
#include <functional>
#include <type_traits> // std::plus and friends, for the table's eval column
#include <utility>

namespace ddx::rt {

namespace detail {

template <typename Fn>
inline constexpr bool is_field_op_v = boost::mp11::mp_set_contains<
    boost::mp11::mp_list<std::plus<>, std::multiplies<>, std::divides<>,
                         std::negate<>>,
    Fn>::value;

#define DDX_RT_PROBE1(fn, Op, label, ...)                                      \
  template <impl::Numeric T>                                                   \
  inline constexpr bool probes_##Op =                                          \
      impl::CArithmetic<T> || requires(const T &u) { fn(u); };
DDX_UNARY_MATH_TABLE(DDX_RT_PROBE1)
#undef DDX_RT_PROBE1

// What a binary functor's body reaches for, asked by name rather than by the
// row's spelling: `max(a, b)` is what a caller writes, and extremum_impl never
// calls one -- it compares and takes a midpoint.  A probe that asked for the
// spelling answered false at TaylorDual, and supported() below then handed
// back a zero node with nothing to say it had.
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
// Comparisons need ordering and nothing else; select needs to ask whether the
// condition is zero.
template <impl::Numeric T>
inline constexpr bool probes_Lt =
    impl::CArithmetic<T> || std::totally_ordered<T>;

template <impl::Numeric T> inline constexpr bool probes_Le = probes_Lt<T>;

template <impl::Numeric T>
inline constexpr bool probes_Select =
    impl::CArithmetic<T> || std::equality_comparable<T>;

// sign_impl only compares, so ordering is the whole requirement.
template <impl::Numeric T>
inline constexpr bool probes_Sign =
    impl::CArithmetic<T> || std::totally_ordered<T>;

// Every row instantiates for every T, so an op the scalar lacks has to compile
// to *something*; T{} is that, and supports() below is how a sweep finds out
// before it reads one.
template <typename Fn, impl::Numeric T, bool Ok>
[[nodiscard]] constexpr T
supported([[maybe_unused]] const impl::Numeric auto &...args) noexcept {
  if constexpr (Ok) {
    return static_cast<T>(Fn{}(args...));
  } else {
    return T{};
  }
}

// One table row as a type: what to call, whether T can, and with how many
// operands.  A leaf is the row with nothing to call.
template <typename Fn, bool Ok, std::size_t Arity> struct Row {
  using functor = Fn;
  static constexpr bool ok = Ok;
  static constexpr std::size_t arity = Arity;
};
// A field op is computed at every T; the probe only decides the rest.
template <typename Fn, bool Probed, std::size_t Arity>
using row_t = Row<Fn, Probed || is_field_op_v<Fn>, Arity>;

// The one switch over the tables.  Every question asked of an opcode --
// compute it, can T compute it, sweep W lanes of it -- is a visitor over the
// Row it names, so the operation set is spelled once here and cannot drift
// between the askers.
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
    // A call is opaque to every scalar: the interpreter calls the function
    // itself, and supports() saying no is what keeps a Taylor sweep away from
    // one.  A sum is the interpreter's too, and adds at every scalar.
#define DDX_RT_DISPATCH(fn, Op, label, arity, kind)                            \
  case OpCode::Op:                                                             \
    return visit(Row<void, Kind::kind == Kind::Nary, arity>{});
    DDX_RT_EXTERN_TABLE(DDX_RT_DISPATCH)
    DDX_RT_NARY_TABLE(DDX_RT_DISPATCH)
#undef DDX_RT_DISPATCH
    // A tensor row is the interpreter's as well, through its kernel table, and
    // at no scalar but the graph's own.
#define DDX_RT_DISPATCH(fn, Op, label, arity, kind)                            \
  case OpCode::Op:                                                             \
    return visit(Row<void, false, arity>{});
    DDX_RT_TENSOR_TABLE(DDX_RT_DISPATCH)
#undef DDX_RT_DISPATCH
  }
  std::unreachable();
}

} // namespace detail

// Whether apply<T>(op, ...) computes `op` or answers T{}.  A leaf is always
// computed.
template <impl::Numeric T>
[[nodiscard]] constexpr bool supports(OpCode op) noexcept {
  return detail::dispatch<T>(op, []<typename R>(R) { return R::ok; });
}

// Templated on the scalar, so the interpreter, constant folding and the builder
// share one definition: at double it computes, at Expression it builds.  The
// operand count is the arity asked for; a row of another arity answers T{},
// as an unsupported one does.
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
