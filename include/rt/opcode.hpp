#pragma once

#include "ops/unary_math.hpp" // DDX_UNARY_MATH_TABLE
#include "util/ranges.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

// The runtime mirror of the compile-time operation set.  Rows are (factory
// spelling, enumerator, label), the shape DDX_UNARY_MATH_TABLE already has.
namespace ddx::rt {

// Every leaf but Const reads an input column.  Only a Var is differentiated:
// being a leaf that is not a Var is what keeps a Param, an Implicit or a Seed
// out of symbols(), out of the reverse sweep's harvest and out of the coupling
// pattern, with no slot-range convention for any of them to agree on.  A Param
// is a named input present in every lane; an Implicit, one the caller solved
// for; a Seed, a seeded product's direction.
#define DDX_RT_LEAF_TABLE(X)                                                   \
  X(constant, Const, "const")                                                  \
  X(var, Var, "var")                                                           \
  X(param, Param, "param")                                                     \
  X(implicit, Implicit, "implicit")                                            \
  X(seed, Seed, "seed")

// The last cell is the ops/adjoints.hpp descriptor carrying the reverse-mode
// rule, which the compile-time ops call too -- the rule is written once, there.
// The eighteen transcendentals get the same treatment from
// DDX_UNARY_MATH_TABLE, whose descriptors are found by name rather than listed.
#define DDX_RT_UNARY_TABLE(X)                                                  \
  X(neg, Neg, "-", std::negate<>, NegateOpFn)                                  \
  X(abs, Abs, "abs", impl::detail::abs_impl, AbsOpFn)                          \
  X(sign, Sign, "sign", impl::detail::sign_impl, SignOpFn)

#define DDX_RT_BINARY_TABLE(X)                                                 \
  X(add, Add, "+", std::plus<>, SumOpFn)                                       \
  X(mul, Mul, "*", std::multiplies<>, MultiplyOpFn)                            \
  X(div, Div, "/", std::divides<>, DivideOpFn)                                 \
  X(pow, Pow, "pow", impl::detail::pow_impl, PowOpFn)                          \
  X(atan2, Atan2, "atan2", impl::detail::atan2_impl, Atan2OpFn)                \
  X(hypot, Hypot, "hypot", impl::detail::hypot_impl, HypotOpFn)                \
  X(max, Max, "max", impl::detail::max_impl, MaxOpFn)                          \
  X(min, Min, "min", impl::detail::min_impl, MinOpFn)

// Their own table, and so no descriptor: the derivative is zero wherever it
// exists, which is what rt/derivative.hpp's `default` already answers.
#define DDX_RT_COMPARE_TABLE(X)                                                \
  X(lt, Lt, "<", impl::detail::lt_impl)                                        \
  X(le, Le, "<=", impl::detail::le_impl)

// The one ternary: `c != 0 ? t : f`, both arms evaluated.
#define DDX_RT_TERNARY_TABLE(X)                                                \
  X(select, Select, "select", impl::detail::select_impl)

#define DDX_RT_OP_TABLE(X)                                                     \
  DDX_RT_LEAF_TABLE(X)                                                         \
  DDX_RT_UNARY_TABLE(X)                                                        \
  DDX_UNARY_MATH_TABLE(X)                                                      \
  DDX_RT_BINARY_TABLE(X)                                                       \
  DDX_RT_COMPARE_TABLE(X)                                                      \
  DDX_RT_TERNARY_TABLE(X)

enum class OpCode : std::uint8_t {
#define DDX_RT_ENUMERATOR(fn, Op, label, ...) Op,
  DDX_RT_OP_TABLE(DDX_RT_ENUMERATOR)
#undef DDX_RT_ENUMERATOR
};

inline constexpr std::size_t op_count = [] {
  std::size_t n = 0;
#define DDX_RT_COUNT(fn, Op, label, ...) ++n;
  DDX_RT_OP_TABLE(DDX_RT_COUNT)
#undef DDX_RT_COUNT
  return n;
}();

namespace detail {

struct OpInfo {
  std::string_view name; // the factory spelling, and the Python name
  std::string_view label;
  std::uint8_t arity;
};

// Arity is which sub-table a row sits in, so the three groups fill one array in
// three passes.  Each row lands at its own enumerator, so nothing depends on
// the table order matching the enum's.
inline constexpr std::array op_info = [] {
  std::array<OpInfo, op_count> t{};
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {#fn, label, 0};
  DDX_RT_LEAF_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {#fn, label, 1};
  DDX_RT_UNARY_TABLE(DDX_RT_ROW)
  DDX_UNARY_MATH_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {#fn, label, 2};
  DDX_RT_BINARY_TABLE(DDX_RT_ROW)
  DDX_RT_COMPARE_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {#fn, label, 3};
  DDX_RT_TERNARY_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
  return t;
}();

// A duplicate would make opcode_of() ambiguous, and the compile-time ops
// pick their enumerator by label.
static_assert([] {
  auto labels = op_info;
  std::ranges::sort(labels, {}, &OpInfo::label);
  return std::ranges::adjacent_find(labels, {}, &OpInfo::label) == labels.end();
}());

} // namespace detail

namespace detail {
// An OpCode names a row: a byte from outside -- a file, a Python int -- is
// range-checked where it is cast, never here.
[[nodiscard]] constexpr const OpInfo &info(OpCode op) noexcept {
  const auto i = static_cast<std::size_t>(op);
  assert(i < op_count);
  return op_info[i];
}
} // namespace detail

[[nodiscard]] constexpr std::string_view label_of(OpCode op) noexcept {
  return detail::info(op).label;
}
[[nodiscard]] constexpr std::string_view name_of(OpCode op) noexcept {
  return detail::info(op).name;
}
[[nodiscard]] constexpr std::uint8_t arity_of(OpCode op) noexcept {
  return detail::info(op).arity;
}

// By label, at compile time for the bridge and at run time for a loader: byte
// values are table-order, so appending a transcendental shifts every enumerator
// above it, and a file names its opcodes by label and remaps them on load.
[[nodiscard]] constexpr std::optional<OpCode>
opcode_of(std::string_view label) noexcept {
  const auto row =
      std::ranges::find(detail::op_info, label, &detail::OpInfo::label);
  return row == detail::op_info.end() ? std::nullopt
                                      : std::optional{static_cast<OpCode>(
                                            row - detail::op_info.begin())};
}

[[nodiscard]] inline std::vector<std::string> opcode_labels() {
  return detail::op_info | std::views::transform(&detail::OpInfo::label) |
         impl::to<std::vector<std::string>>();
}

template <impl::Numeric T>
[[nodiscard]] constexpr bool is_commutative(OpCode op) noexcept {
  return (op == OpCode::Mul) ? impl::CCommutativeMultiply<T>
                             : (op == OpCode::Add || op == OpCode::Max ||
                                op == OpCode::Min || op == OpCode::Hypot);
}

[[nodiscard]] constexpr bool is_leaf(OpCode op) noexcept {
  return arity_of(op) == 0;
}

// How many input columns of each kind a call passes, in the order it passes
// them: symbols, parameters, implicits, then a seeded product's direction.
struct Inputs {
  std::size_t vars = 0;
  std::size_t params = 0;
  std::size_t implicits = 0;
  std::size_t seeds = 0;

  [[nodiscard]] constexpr std::size_t named() const noexcept {
    return vars + params + implicits;
  }
  [[nodiscard]] constexpr std::size_t total() const noexcept {
    return named() + seeds;
  }
  constexpr bool operator==(const Inputs &) const = default;
};

// Which column of `xs` a leaf reads.  Written once, because the interpreter,
// codegen and the ABI have to agree.
[[nodiscard]] constexpr std::size_t input_column(const Inputs &in, OpCode op,
                                                 std::uint32_t slot) noexcept {
  switch (op) {
  case OpCode::Var:
    return slot;
  case OpCode::Param:
    return in.vars + slot;
  case OpCode::Implicit:
    return in.vars + in.params + slot;
  default:
    return in.named() + slot;
  }
}

} // namespace ddx::rt
