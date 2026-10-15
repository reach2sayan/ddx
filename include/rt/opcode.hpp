#pragma once

#include "ops/unary_math.hpp" // DDX_UNARY_MATH_TABLE
#include "util/ranges.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The runtime mirror of the compile-time operation set.  Rows are (factory
// spelling, enumerator, label), the shape DDX_UNARY_MATH_TABLE already has.
namespace ddx::rt {

// Every leaf but Const reads an input column.  Only a Var is differentiated:
// being a leaf that is not a Var is what keeps a Param, an Implicit or a Seed
// out of symbols(), out of the reverse sweep's harvest and out of the coupling
// pattern, with no slot-range convention for any of them to agree on.  A Param
// is a named input present in every lane; an Implicit, one the caller solved
// for; a Seed, a seeded product's direction.  The last cell is what the slot
// is: nothing, a place among the named inputs of that kind, or a position.
// The named kinds are their own table, in the order a point passes them: it
// is what the Builder keeps one name list per and what var()'s spellings,
// the Python factories and the column layout are generated from.
#define DDX_RT_NAMED_TABLE(X, ...)                                             \
  X(var, Var, "var" __VA_OPT__(, ) __VA_ARGS__)                                \
  X(param, Param, "param" __VA_OPT__(, ) __VA_ARGS__)                          \
  X(implicit, Implicit, "implicit" __VA_OPT__(, ) __VA_ARGS__)
#define DDX_RT_LEAF_TABLE(X)                                                   \
  X(constant, Const, "const", Literal)                                         \
  DDX_RT_NAMED_TABLE(X, Named)                                                 \
  X(seed, Seed, "seed", Positional)

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

// A function the caller supplies with its own partials, of any number of
// arguments.  Extern (the value) and ExternD (one partial -- which, and of
// which function, packed into the slot) take one operand: the argument list.
// A list of one is the argument itself; a longer one is ExternArgs(first,
// rest), the same rule all the way down, so every node holds only real
// operands.  The last cells are the arity and the Kind.
#define DDX_RT_EXTERN_TABLE(X)                                                 \
  X(extern_call, Extern, "extern", 1, Extern)                                  \
  X(extern_partial, ExternD, "extern_d", 1, Extern)                            \
  X(extern_args, ExternArgs, "extern_args", 2, Args)

#define DDX_RT_OP_TABLE(X)                                                     \
  DDX_RT_LEAF_TABLE(X)                                                         \
  DDX_RT_UNARY_TABLE(X)                                                        \
  DDX_UNARY_MATH_TABLE(X)                                                      \
  DDX_RT_BINARY_TABLE(X)                                                       \
  DDX_RT_COMPARE_TABLE(X)                                                      \
  DDX_RT_TERNARY_TABLE(X)                                                      \
  DDX_RT_EXTERN_TABLE(X)

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

// What a walker does with a node: the one classification every sweep, emitter
// and checker switches on, so none tests an opcode ahead of its arity.  A
// Const is a literal, an Input reads a column, an Extern calls the program's
// function (its value or one partial), an Args node is an argument list.
enum class Kind : std::uint8_t {
  Const,
  Input,
  Unary,
  Binary,
  Ternary,
  Extern,
  Args
};

namespace detail {

enum class Role : std::uint8_t { Literal, Named, Positional };

struct OpInfo {
  std::string_view name; // the factory spelling, and the Python name
  std::string_view label;
  std::uint8_t arity;
  Kind kind;
};

// Arity and kind are which sub-table a row sits in, so the groups fill one
// array in one pass each.  Each row lands at its own enumerator, so nothing
// depends on the table order matching the enum's.
inline constexpr std::array op_info = [] {
  std::array<OpInfo, op_count> t{};
#define DDX_RT_ROW(fn, Op, label, role, ...)                                   \
  t[static_cast<std::size_t>(OpCode::Op)] = {                                  \
      #fn, label, 0,                                                           \
      Role::role == Role::Literal ? Kind::Const : Kind::Input};
  DDX_RT_LEAF_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {#fn, label, 1, Kind::Unary};
  DDX_RT_UNARY_TABLE(DDX_RT_ROW)
  DDX_UNARY_MATH_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {#fn, label, 2, Kind::Binary};
  DDX_RT_BINARY_TABLE(DDX_RT_ROW)
  DDX_RT_COMPARE_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {#fn, label, 3, Kind::Ternary};
  DDX_RT_TERNARY_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, arity, kind)                                 \
  t[static_cast<std::size_t>(OpCode::Op)] = {#fn, label, arity, Kind::kind};
  DDX_RT_EXTERN_TABLE(DDX_RT_ROW)
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
[[nodiscard]] constexpr Kind kind_of(OpCode op) noexcept {
  return detail::info(op).kind;
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
  return kind_of(op) == Kind::Const || kind_of(op) == Kind::Input;
}
[[nodiscard]] constexpr bool is_extern(OpCode op) noexcept {
  return kind_of(op) == Kind::Extern || kind_of(op) == Kind::Args;
}
// What text and Python may call by name: an operation on operands, not a
// leaf and not a function the program registered.
[[nodiscard]] constexpr bool callable(OpCode op) noexcept {
  return !is_leaf(op) && !is_extern(op);
}

// Which kind of named input a leaf is, in table order.
[[nodiscard]] constexpr std::optional<std::size_t>
named_index(OpCode op) noexcept {
  std::size_t i = 0;
#define DDX_RT_NAMED(fn, Op, label)                                            \
  if (op == OpCode::Op) {                                                      \
    return i;                                                                  \
  }                                                                            \
  ++i;
  DDX_RT_NAMED_TABLE(DDX_RT_NAMED)
#undef DDX_RT_NAMED
  return std::nullopt;
}
inline constexpr std::size_t named_count = [] {
  std::size_t n = 0;
#define DDX_RT_NAMED(...) ++n;
  DDX_RT_NAMED_TABLE(DDX_RT_NAMED)
#undef DDX_RT_NAMED
  return n;
}();

// An extern node's slot: which registered function, and which component of
// its answer -- 0 the value, 1..k the first partials, then the upper triangle
// of the second, row-major.  A half-word each; fits() is the one bound.
struct ExternSlot {
  std::uint32_t function = 0;
  std::uint32_t component = 0;
  static constexpr std::uint32_t limit = 1u << 16;

  [[nodiscard]] static constexpr ExternSlot of(std::uint32_t slot) noexcept {
    return {.function = slot >> 16, .component = slot & (limit - 1)};
  }
  [[nodiscard]] constexpr std::uint32_t packed() const noexcept {
    return (function << 16) | component;
  }
  [[nodiscard]] static constexpr bool fits(std::size_t function,
                                           std::size_t components) noexcept {
    return function < limit && components <= limit;
  }
};


// How many input columns of each kind a call passes, in the order it passes
// them: the named kinds in table order, then a seeded product's direction.
struct Inputs {
  std::array<std::size_t, named_count> by_kind{};
  std::size_t seeds = 0;

  [[nodiscard]] constexpr std::size_t count(OpCode op) const noexcept {
    return by_kind[*named_index(op)];
  }
  [[nodiscard]] constexpr std::size_t named() const noexcept {
    return std::ranges::fold_left(by_kind, 0uz, std::plus<>{});
  }
  [[nodiscard]] constexpr std::size_t total() const noexcept {
    return named() + seeds;
  }
  constexpr bool operator==(const Inputs &) const = default;
};

// Which column of `xs` a leaf reads: its kind's columns start after every
// kind before it.  Written once, because the interpreter, codegen and the ABI
// have to agree.
[[nodiscard]] constexpr std::size_t input_column(const Inputs &in, OpCode op,
                                                 std::uint32_t slot) noexcept {
  const auto before = named_index(op).value_or(named_count);
  return std::ranges::fold_left(std::span{in.by_kind}.first(before), 0uz,
                                std::plus<>{}) +
         slot;
}

} // namespace ddx::rt
