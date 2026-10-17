#pragma once

#include "ops/unary_math.hpp" // DDX_UNARY_MATH_TABLE
#include "util/ranges.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
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
#define DDX_RT_NAMED_TABLE(X)                                                  \
  X(var, Var, "var", Named)                                                    \
  X(param, Param, "param", Named)                                              \
  X(implicit, Implicit, "implicit", Named)
#define DDX_RT_LEAF_TABLE(X)                                                   \
  X(constant, Const, "const", Literal)                                         \
  DDX_RT_NAMED_TABLE(X)                                                        \
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

// A node over a list of ids rather than over operands: `a` is where the list
// starts in Model::lists (has_list below), its length the function's arity
// for an extern, else the slot.  The last cells are the arity and the Kind.
//
// A function the caller supplies with its own partials, of any number of
// arguments: Extern is the value, ExternD one partial -- which, and of which
// function, packed into the slot.
#define DDX_RT_EXTERN_TABLE(X)                                                 \
  X(extern_call, Extern, "extern", 1, Extern)                                  \
  X(extern_partial, ExternD, "extern_d", 1, Extern)
// A sum of `slot` terms in list order: what a chain of += builds, one node
// where a chain of Add would be one per term.  Every product term is fused
// into the running sum as an Add fuses its one (contraction_at), so a sum of
// products is a loop of fma.
#define DDX_RT_NARY_TABLE(X) X(sum_n, SumN, "sum_n", 1, Nary)

// A node whose value is a vector of `width` scalars per point, or a scalar
// read off one.  Operands are ids like any other; the slot is what the row
// says -- no node carries data, and a shape is a function of the source
// (width_of in rt/builder.hpp), never of the node alone:
//   VecConst, MatConst  slot = index into Model::tensors; the data lives there
//   Pack                a = list of `slot` scalars
//   Gather              a = vec, slot = element; a scalar
//   Map, Zip            a (, b) = vecs, slot = the elementwise opcode
//   Scale               a = scalar, b = vec
//   Sum, Dot            a (, b) = vecs; a scalar
//   Gemv                a = MatConst, b = vec, slot = 1 for M^T v
//   AddAt               a = vec, b = scalar, slot = element: a + b e_slot
// A matrix is a row too, column-major, its shape derived from its operands
// (shape_of in rt/builder.hpp); the dense Hessian sweep forms these:
//   MatRow              a = MatConst, slot = 1 for M^T: the table as a row
//   Diag                a = vec w: the w x w diagonal
//   Outer               a = vec r, b = vec c: a b^T
//   RowScale            a = vec r, b = matrix r x c: diag(a) M
//   ColScale            a = matrix r x c, b = vec c: M diag(b)
//   MatVec              a = matrix, b = vec, slot = 1 for M^T v
//   MatMul              a = matrix r x k, b = matrix k x c
//   StackRows           a = list of `slot` vectors, each a row
//   RowOf               a = matrix, slot = row: that row, as a vector
// The last cells are the arity and the Kind.
#define DDX_RT_TENSOR_TABLE(X)                                                 \
  X(vec_const, VecConst, "vconst", 0, Tensor)                                  \
  X(mat_const, MatConst, "mconst", 0, Tensor)                                  \
  X(pack, Pack, "pack", 1, Tensor)                                             \
  X(gather, Gather, "gather", 1, Tensor)                                       \
  X(vmap, Map, "vmap", 1, Tensor)                                              \
  X(vzip, Zip, "vzip", 2, Tensor)                                              \
  X(vscale, Scale, "vscale", 2, Tensor)                                        \
  X(vsum, Sum, "vsum", 1, Tensor)                                              \
  X(vdot, Dot, "vdot", 2, Tensor)                                              \
  X(gemv, Gemv, "gemv", 2, Tensor)                                             \
  X(add_at, AddAt, "add_at", 2, Tensor)                                        \
  X(mat_row, MatRow, "mrow", 1, Tensor)                                        \
  X(diag, Diag, "diag", 1, Tensor)                                             \
  X(outer, Outer, "outer", 2, Tensor)                                          \
  X(row_scale, RowScale, "row_scale", 2, Tensor)                               \
  X(col_scale, ColScale, "col_scale", 2, Tensor)                               \
  X(matvec, MatVec, "matvec", 2, Tensor)                                       \
  X(matmul, MatMul, "matmul", 2, Tensor)                                       \
  X(stack_rows, StackRows, "stack_rows", 1, Tensor)                            \
  X(row_of, RowOf, "row_of", 1, Tensor)

#define DDX_RT_OP_TABLE(X)                                                     \
  DDX_RT_LEAF_TABLE(X)                                                         \
  DDX_RT_UNARY_TABLE(X)                                                        \
  DDX_UNARY_MATH_TABLE(X)                                                      \
  DDX_RT_BINARY_TABLE(X)                                                       \
  DDX_RT_COMPARE_TABLE(X)                                                      \
  DDX_RT_TERNARY_TABLE(X)                                                      \
  DDX_RT_EXTERN_TABLE(X)                                                       \
  DDX_RT_NARY_TABLE(X)                                                         \
  DDX_RT_TENSOR_TABLE(X)

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
// function (its value or one partial), a Nary node reduces a list of any
// length, a Tensor node computes over or into a vector row
// (DDX_RT_TENSOR_TABLE).
enum class Kind : std::uint8_t {
  Const,
  Input,
  Unary,
  Binary,
  Ternary,
  Extern,
  Nary,
  Tensor
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
      #fn, label, 0, Role::role == Role::Literal ? Kind::Const : Kind::Input};
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
  DDX_RT_NARY_TABLE(DDX_RT_ROW)
  DDX_RT_TENSOR_TABLE(DDX_RT_ROW)
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

// The leaf table opens the op table, so a leaf is one compare: the sweep asks
// on every step, and a kind_of() here is a table load ahead of its dispatch.
[[nodiscard]] constexpr bool is_leaf(OpCode op) noexcept {
  return op <= OpCode::Seed;
}
static_assert(std::ranges::all_of(std::views::iota(0uz, op_count), [](auto i) {
  const auto op = static_cast<OpCode>(i);
  return is_leaf(op) ==
         (kind_of(op) == Kind::Const || kind_of(op) == Kind::Input);
}));
[[nodiscard]] constexpr bool is_extern(OpCode op) noexcept {
  return kind_of(op) == Kind::Extern;
}
// Whether `a` is where the node's list starts in Model::lists rather than an
// operand: an extern's arguments, a Pack's or StackRows' items, a sum's terms.
[[nodiscard]] constexpr bool has_list(OpCode op) noexcept {
  return is_extern(op) || kind_of(op) == Kind::Nary || op == OpCode::Pack ||
         op == OpCode::StackRows;
}
// What text and Python may call by name: a scalar operation on operands, not
// a leaf, not a function the program registered, not a list and not a tensor
// row.
[[nodiscard]] constexpr bool callable(OpCode op) noexcept {
  return !is_leaf(op) && !has_list(op) && kind_of(op) != Kind::Tensor;
}
// Whether a node's value is a vector row rather than a scalar: the tensor ops
// that read vectors into a scalar -- Gather, Sum, Dot -- are not, and MatConst
// is a table a Gemv reads, not a row anything sweeps.
[[nodiscard]] constexpr bool tensor_valued(OpCode op) noexcept {
  return kind_of(op) == Kind::Tensor && op != OpCode::Gather &&
         op != OpCode::Sum && op != OpCode::Dot && op != OpCode::MatConst;
}

// The dense Hessian's matrix rows, the tensor table's tail: built under no
// root, so no sweep differentiates through them.
[[nodiscard]] constexpr bool is_matrix_row(OpCode op) noexcept {
  return op >= OpCode::MatRow;
}
static_assert(std::to_underlying(OpCode::RowOf) + 1 == op_count);

// Which kind of named input a leaf is, in table order.
[[nodiscard]] constexpr std::optional<std::size_t>
named_index(OpCode op) noexcept {
  std::size_t i = 0;
#define DDX_RT_NAMED(fn, Op, ...)                                              \
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

// One T per named kind: a std::array, and on every compiler but one nothing
// besides.  GCC before 15 deallocates twice when constant evaluation copies an
// array whose elements allocate, so there the copies are spelled element by
// element -- the one spelling it gets right.  An equation copies the arena it
// was lent, which is what makes a consteval equation reach this at all; the
// run-time copy is the array's own either way.
template <typename T> struct PerKind : std::array<T, named_count> {
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 15
  using Array = std::array<T, named_count>;
  constexpr PerKind() = default;
  constexpr PerKind(const PerKind &other) : Array(other.copied()) {
    if consteval {
      std::ranges::copy(other, this->begin());
    }
  }
  constexpr PerKind(PerKind &&other) noexcept
      : Array(std::move(other).taken()) {
    if consteval {
      std::ranges::move(other, this->begin());
    }
  }
  constexpr PerKind &operator=(const PerKind &other) {
    if consteval {
      std::ranges::copy(other, this->begin());
    } else {
      static_cast<Array &>(*this) = other;
    }
    return *this;
  }
  constexpr PerKind &operator=(PerKind &&other) noexcept {
    if consteval {
      std::ranges::move(other, this->begin());
    } else {
      static_cast<Array &>(*this) = std::move(other);
    }
    return *this;
  }

private:
  // What the base is built from: its own copy, which is the whole of this at
  // run time, and an empty array during constant evaluation -- the elements
  // the constructor's body then assigns are what the bug is dodged by.  A
  // prvalue, so the run-time copy lands in the base and no second one exists.
  [[nodiscard]] constexpr Array copied() const {
    if consteval {
      return Array{};
    }
    return *this;
  }
  [[nodiscard]] constexpr Array taken() && {
    if consteval {
      return Array{};
    }
    return std::move(*this);
  }
#endif
};

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
  // A scan of fixed length: a fold over a prefix of variable length
  // vectorises, and the sweep's leaf step pays for the loop.
  std::array<std::size_t, named_count + 1> start{};
  std::inclusive_scan(in.by_kind.begin(), in.by_kind.end(), start.begin() + 1);
  return start[named_index(op).value_or(named_count)] + slot;
}

} // namespace ddx::rt
