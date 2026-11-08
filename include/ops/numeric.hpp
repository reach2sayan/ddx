#pragma once
// The scalar concepts the library is written against. Numeric admits a double,
// a dual, a matrix or a quaternion alike.

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

// Closed under the four arithmetic operators and negation, and constructible
// from int because differentiation creates the constants 0 and 1.
template <typename T>
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

// Declared rather than deduced, since Numeric admits matrices and quaternions;
// only arithmetic types commute by default.
template <typename T>
inline constexpr bool is_commutative_multiply_v = CArithmetic<T>;

template <typename T>
concept CCommutativeMultiply =
    is_commutative_multiply_v<std::remove_cvref_t<T>>;

template <typename O>
concept COperation =
    requires { typename O::value_type; } && Numeric<typename O::value_type>;

template <typename T> inline constexpr bool is_expression_type_v = false;
template <typename T>
concept CExpression = is_expression_type_v<std::remove_cvref_t<T>>;

template <COperation Op, CExpression... Children> class Expression;

// Expression's storage base, which holds operands unless they are all empty.
template <bool Stateless, COperation Op, CExpression... Children>
class ExpressionImpl;

template <COperation Op, CExpression Exp>
using MonoExpression = Expression<Op, Exp>;

template <Numeric T, auto... V> class Lit;
template <Numeric T> using Constant = Lit<T>;

namespace detail {
template <CArithmetic T> consteval T compile_time_factorial(T Order) {
  return std::ranges::fold_left(std::views::iota(T{1}, T(Order + 1)), T{1},
                                std::multiplies{});
}
static_assert(compile_time_factorial(3) == 6);
static_assert(compile_time_factorial(5) == 120);
static_assert(compile_time_factorial(7) == 5040);
} // namespace detail

} // namespace ddx::impl
