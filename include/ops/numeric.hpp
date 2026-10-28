#pragma once
// The scalar vocabulary the whole library is written against.  Nothing here
// knows what an expression tree is: `Numeric` admits a double, a dual, a matrix
// or a quaternion alike, which is what lets the compile-time graph, the runtime
// graph and the JIT share one set of operation descriptors.

#include "util/config.hpp"

#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>

namespace ddx::impl {

template <typename T>
concept CArithmetic = std::integral<std::remove_cvref_t<T>> ||
                      std::floating_point<std::remove_cvref_t<T>>;

// Closed under the four arithmetic operators and negation.
template <typename T>
// constructible_from<T, int>: differentiation manufactures exactly 0 and 1.
concept CFieldLike = std::default_initializable<T> &&
                     std::constructible_from<T, int> && requires(T a, T b) {
                       { a + b } -> std::convertible_to<T>;
                       { a - b } -> std::convertible_to<T>;
                       { a * b } -> std::convertible_to<T>;
                       { a / b } -> std::convertible_to<T>;
                       { -a } -> std::convertible_to<T>;
                     };

template <typename T>
concept Numeric = CArithmetic<T> || CFieldLike<T>;

// Declared, not deduced: Numeric admits matrices and quaternions, and the
// default is the safe answer.
template <typename T>
inline constexpr bool is_commutative_multiply_v = CArithmetic<T>;

template <typename T>
concept CCommutativeMultiply =
    is_commutative_multiply_v<std::remove_cvref_t<T>>;

// Decomposable into {value, derivative} via std::tuple_size / get<I>.
template <typename T>
concept CTupleLike =
    requires { std::tuple_size<std::remove_cvref_t<T>>::value; };

namespace detail {
template <typename F, typename Seq>
inline constexpr bool index_invocable_v = false;
template <typename F, std::size_t... Is>
inline constexpr bool index_invocable_v<F, std::index_sequence<Is...>> =
    (requires(F &&f) { DDX_FWD(f).template operator()<Is>(); } && ...);

template <typename F, typename Seq>
inline constexpr bool pack_invocable_v = false;
template <typename F, std::size_t... Is>
inline constexpr bool pack_invocable_v<F, std::index_sequence<Is...>> =
    requires(F &&f) { DDX_FWD(f).template operator()<Is...>(); };
} // namespace detail

// static_for  : hands over one index at a time, for a fold of statements;
// index_apply : hands over the whole pack at once,
template <std::size_t N>
constexpr void static_for(auto &&f) noexcept
  requires detail::index_invocable_v<decltype(f), std::make_index_sequence<N>>
{
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (DDX_FWD(f).template operator()<Is>(), ...);
  }(std::make_index_sequence<N>{});
}

template <std::size_t N>
[[nodiscard]] constexpr decltype(auto) index_apply(auto &&f)
  requires detail::pack_invocable_v<decltype(f), std::make_index_sequence<N>>
{
  return [&]<std::size_t... Is>(std::index_sequence<Is...>) -> decltype(auto) {
    return DDX_FWD(f).template operator()<Is...>();
  }(std::make_index_sequence<N>{});
}

template <typename O>
concept COperation =
    requires { typename O::value_type; } && Numeric<typename O::value_type>;

template <typename T> inline constexpr bool is_expression_type_v = false;
template <typename T>
concept CExpression = is_expression_type_v<std::remove_cvref_t<T>>;

template <COperation Op, CExpression... Children> class Expression;

// Expression's storage base, and the CRTP `Derived`.
template <bool Stateless, COperation Op, CExpression... Children>
class ExpressionImpl;

template <COperation Op, CExpression Exp>
using MonoExpression = Expression<Op, Exp>;

template <Numeric T, auto... V> class Lit;
template <Numeric T> using Constant = Lit<T>;

namespace detail {
// Same fold as binomial() in md/layouts.hpp.
template <CArithmetic T> consteval T compile_time_factorial(T Order) {
  return std::ranges::fold_left(std::views::iota(T{1}, T(Order + 1)), T{1},
                                std::multiplies{});
}
static_assert(compile_time_factorial(3) == 6);
static_assert(compile_time_factorial(5) == 120);
static_assert(compile_time_factorial(7) == 5040);
} // namespace detail

} // namespace ddx::impl
