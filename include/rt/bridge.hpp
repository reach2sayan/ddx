#pragma once

#include "ops/operations.hpp"
#include "rt/apply.hpp" // supports
#include "rt/expressions.hpp"
#include "symbolic/expressions.hpp"
#include "symbolic/traits.hpp"

#include <array>
#include <tuple>
#include <type_traits>

// Lowers a compile-time expression tree into the runtime graph by a
// compile-time recursion over its type.
namespace ddx::rt {

namespace detail {

// Sum and Multiply are n-ary in the type but binary in the graph, so a wide
// node is folded left into a chain of binary nodes.
template <impl::Numeric T, impl::CExpression U>
[[nodiscard]] constexpr Expression<T> lower(Arena<T> &b, const U &e) {
  // Literals in a dual-valued tree are duals, so val() strips them to the base
  // scalar. The call is unqualified so ADL can find the dual overload.
  using impl::val;
  if constexpr (impl::CVariable<U>) {
    return var(b, U::label.view());
  } else if constexpr (impl::CLit<U>) {
    return lit(b, static_cast<T>(val(U::value)));
  } else if constexpr (impl::CConstant<U>) {
    return lit(b,
               static_cast<T>(val(e.template eval_seeded<impl::mp::mp_list<>>(
                   std::array<T, 1>{}))));
  } else {
    constexpr auto code = opcode_of(U::op_type::label);
    static_assert(code.has_value(), "no OpCode row carries this op's label");
    return std::apply(
        [&](const auto &first, const auto &...rest) {
          if constexpr (sizeof...(rest) == 0) {
            return Expression<T>::form(*code, lower(b, first));
          } else if constexpr (*code == OpCode::Select) {
            // Brace initialisation guarantees left-to-right lowering order.
            const std::array arms{lower(b, first), lower(b, rest)...};
            return Expression<T>::form(*code, arms[0], arms[1], arms[2]);
          } else {
            Expression acc = lower(b, first);
            ((acc = Expression<T>::form(*code, acc, lower(b, rest))), ...);
            return acc;
          }
        },
        e.expressions());
  }
}

// Whether every op in the tree is supported at scalar U. Sweeps check first,
// since apply() silently returns U{} for an unsupported op.
template <impl::Numeric U, impl::CExpression E>
[[nodiscard]] consteval bool computes_at_tree() {
  using Tree = std::remove_cvref_t<E>;
  if constexpr (impl::CExpressionNode<Tree>) {
    constexpr auto code = opcode_of(Tree::op_type::label);
    static_assert(code.has_value(), "no OpCode row carries this op's label");
    return supports<U>(*code) &&
           []<typename... C>(std::type_identity<std::tuple<C...>>) {
             return (computes_at_tree<U, C>() && ...);
           }(std::type_identity<typename Tree::children_t>{});
  } else {
    return true;
  }
}

} // namespace detail

// Symbols are declared first in the compile-time Equation's order so their
// columns match. Trees with frozen leaves are refused, as a Var is always live.
template <impl::Numeric T, impl::CExpression U>
  requires(!impl::holds_frozen<U>())
[[nodiscard]] constexpr Expression<T> to_graph(Arena<T> &b, const U &e) {
  impl::mp::mp_for_each<impl::extract_symbols_from_expr_t<U>>(
      [&b]<typename S>(S) { (void)var(b, S::name); });
  return detail::lower(b, e);
}

} // namespace ddx::rt
