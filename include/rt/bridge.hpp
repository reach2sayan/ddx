#pragma once

#include "ops/operations.hpp"
#include "rt/apply.hpp" // supports
#include "rt/expressions.hpp"
#include "symbolic/expressions.hpp"
#include "symbolic/traits.hpp"

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

// Lowering a compile-time tree into the runtime graph: the structure is in the
// type, so the walk is a compile-time recursion that emits nodes.
namespace ddx::rt {

namespace detail {

// Sum and Multiply are n-ary in the type but binary in the graph, so a wide
// node folds left into a chain.
template <impl::Numeric T, impl::CExpression U>
[[nodiscard]] constexpr Expression<T> lower(Arena<T> &b, const U &e) {
  // A dual-valued tree's literals and constants are duals; the graph is over
  // the base.  Unqualified val(): ADL finds the dual's, ordinary lookup the
  // scalar's.
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
          } else {
            Expression acc = lower(b, first);
            ((acc = Expression<T>::form(*code, acc, lower(b, rest))), ...);
            return acc;
          }
        },
        e.expressions());
  }
}

// Whether every op the tree holds computes at U.  A sweep at a Taylor or dual
// scalar asks before it runs: apply() answers U{} for an op the scalar lacks,
// which is a wrong number with nothing to say so.
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

// The symbols are declared first, in the compile-time Equation's canonical
// order, so a symbol's graph column is its column there too.  A frozen leaf has
// no spelling here: a Var is always differentiated, so lowering one would
// quietly revive what the caller held -- the tree is refused instead.
template <impl::Numeric T, impl::CExpression U>
  requires(!impl::holds_frozen<U>())
[[nodiscard]] constexpr Expression<T> to_graph(Arena<T> &b, const U &e) {
  impl::mp::mp_for_each<impl::extract_symbols_from_expr_t<U>>(
      [&b]<typename S>(S) { (void)var(b, S::name); });
  return detail::lower(b, e);
}

} // namespace ddx::rt
