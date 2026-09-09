#pragma once

#include "ops/operations.hpp"
#include "rt/expressions.hpp"
#include "symbolic/expressions.hpp"
#include "symbolic/traits.hpp"

#include <array>
#include <tuple>
#include <utility>

// Lowering a compile-time tree into the runtime graph: the structure is in the
// type, so the walk is a compile-time recursion that emits nodes.
namespace ddx::rt {

// Sum and Multiply are n-ary in the type but binary in the graph, so a wide
// node folds left into a chain.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T>
to_graph(Builder<T> &b, const impl::CExpression auto &e) {
  using U = std::remove_cvref_t<decltype(e)>;
  if constexpr (impl::CVariable<U>) {
    return var(b, U::label.view());
  } else if constexpr (impl::CLit<U>) {
    return lit(b, static_cast<T>(U::value));
  } else if constexpr (impl::CConstant<U>) {
    return lit(b,
               static_cast<double>(e.template eval_seeded<impl::mp::mp_list<>>(
                   std::array<double, 1>{})));
  } else {
    constexpr auto code = opcode_of(U::op_type::label);
    static_assert(code.has_value(), "no OpCode row carries this op's label");
    return std::apply(
        [&](const auto &first, const auto &...rest) {
          if constexpr (sizeof...(rest) == 0) {
            return RTExpression<T>::form(*code, to_graph(b, first));
          } else {
            RTExpression acc = to_graph(b, first);
            ((acc = RTExpression<T>::form(*code, acc, to_graph(b, rest))), ...);
            return acc;
          }
        },
        e.expressions());
  }
}

} // namespace ddx::rt
