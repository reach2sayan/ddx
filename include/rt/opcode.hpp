#pragma once

#include "ops/unary_math.hpp" // DDX_UNARY_MATH_TABLE
#include "util/ranges.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// The runtime mirror of the compile-time operation set.  Rows are (factory
// spelling, enumerator, label), the shape DDX_UNARY_MATH_TABLE already has.
namespace ddx::rt {

// Every leaf but Const reads an input column.  Only a Var is differentiated:
// being a leaf that is not a Var is what keeps a Param, an Implicit or a Seed
// out of symbols(), out of the reverse sweep's harvest and out of the coupling
// pattern, with no slot-range convention for any of them to agree on.  A Param
// is a named input present in every Want's graph; an Implicit, one the caller
// solved for; a Seed, a seeded product's direction.  The last cell is what the
// index is: nothing, a place among the named inputs of that kind, or a
// position. The named kinds are their own table, in the order a point passes
// them: it is what the Arena keeps one name list per and what var()'s
// spellings, the Python factories and the column layout are generated from.
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
  DDX_BINARY_MATH_TABLE(X)

// Their own table, and so no descriptor: the derivative is zero wherever it
// exists, which is what rt/derivative.hpp's `default` already answers.
#define DDX_RT_COMPARE_TABLE(X)                                                \
  X(lt, Lt, "<", impl::detail::lt_impl)                                        \
  X(le, Le, "<=", impl::detail::le_impl)

// `c != 0 ? t : f`, both arms evaluated.  VSelect is the one over tensor arms
// of one shape, the condition a scalar or of theirs, where the arm taken is
// copied, never scaled by 1 or 0.  The last cell is what the node yields.
#define DDX_RT_TERNARY_TABLE(X)                                                \
  X(select, Select, "select", impl::detail::select_impl, Scalar)               \
  X(vselect, VSelect, "vselect", impl::detail::select_impl, Row)

// A node over a list of ids rather than over operands: `a` is where the list
// starts in Model::lists (has_list below), its length the function's arity
// for an extern, else the Count it carries.  The last cells are the arity and
// the Kind.
//
// A function the caller supplies with its own partials, of any number of
// arguments: Extern is the value, ExternD one partial -- which, and of which
// function, its ExternRef.
#define DDX_RT_EXTERN_TABLE(X)                                                 \
  X(extern_call, Extern, "extern", 1, Extern)                                  \
  X(extern_partial, ExternD, "extern_d", 1, Extern)
// A left fold of `Count` terms in list order, ((t0 op t1) op t2) ..., the op
// the last cell: what a chain of +=, *= or /= builds, one node where a chain
// of the binary op would be one per term.  In a sum every product term is
// fused into the running sum as an Add fuses its one (contraction_at), so a
// sum of products is a loop of fma.
#define DDX_RT_NARY_TABLE(X)                                                   \
  X(sum_n, SumN, "sum_n", 1, Nary, Add)                                        \
  X(prod_n, ProdN, "prod_n", 1, Nary, Mul)                                     \
  X(quot_n, QuotN, "quot_n", 1, Nary, Div)

// A node whose value is a vector of `width` scalars per point, or a scalar
// read off one.  Operands are ids like any other; no node carries data, and a
// shape is a function of the source (shape_of in rt/arena.hpp), never of the
// node alone.  A matrix is a row too, column-major.  Map, Zip, Scale, Sum and
// Dot read one as the vector of its elements.
//   VecConst, MatConst  the data lives in Model::tensors
//   Pack                a = list of scalars
//   Gather              a = vec; the element, a scalar
//   Map, Zip            a (, b) = vecs; the elementwise opcode
//   Scale               a = scalar, b = vec
//   Sum, Dot            a (, b) = vecs; a scalar
//   AddAt               a = vec, b = scalar: a + b e_element
//   MatRow              a = MatConst, the table as a row; in the dense sweep
//                       also a = matrix
//   Outer               a = vec r, b = vec c: a b^T
//   MatVec              a = matrix, b = vec
//   Diag                a = vec w: the w x w diagonal
//   RowScale            a = vec r, b = matrix r x c: diag(a) M
//   ColScale            a = matrix r x c, b = vec c: M diag(b)
//   MatMul              a = matrix r x k, b = matrix k x c
//   StackRows           a = list of vectors, each a row
//   RowOf               a = matrix; that row, as a vector
// The last cells are the arity, the Kind, the payload and what it yields.
#define DDX_RT_TENSOR_TABLE(X)                                                 \
  X(vec_const, VecConst, "vconst", 0, Tensor, TensorRef, Row)                  \
  X(mat_const, MatConst, "mconst", 0, Tensor, TensorRef, Table)                \
  X(pack, Pack, "pack", 1, Tensor, Count, Row)                                 \
  X(gather, Gather, "gather", 1, Tensor, Element, Scalar)                      \
  X(vmap, Map, "vmap", 1, Tensor, OpCode, Row)                                 \
  X(vzip, Zip, "vzip", 2, Tensor, OpCode, Row)                                 \
  X(vscale, Scale, "vscale", 2, Tensor, NoPayload, Row)                        \
  X(vsum, Sum, "vsum", 1, Tensor, NoPayload, Scalar)                           \
  X(vdot, Dot, "vdot", 2, Tensor, NoPayload, Scalar)                           \
  X(add_at, AddAt, "add_at", 2, Tensor, Element, Row)                          \
  X(mat_row, MatRow, "mrow", 1, Tensor, Orientation, Row)                      \
  X(diag, Diag, "diag", 1, Tensor, NoPayload, Row)                             \
  X(outer, Outer, "outer", 2, Tensor, NoPayload, Row)                          \
  X(row_scale, RowScale, "row_scale", 2, Tensor, NoPayload, Row)               \
  X(col_scale, ColScale, "col_scale", 2, Tensor, NoPayload, Row)               \
  X(matvec, MatVec, "matvec", 2, Tensor, Orientation, Row)                     \
  X(matmul, MatMul, "matmul", 2, Tensor, NoPayload, Row)                       \
  X(stack_rows, StackRows, "stack_rows", 1, Tensor, Count, Row)                \
  X(row_of, RowOf, "row_of", 1, Tensor, Element, Row)

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

// What a node carries besides its operands, one type per meaning; each row
// of the table names the one its op carries, and a node's word is read back
// only as that type (detail::get in rt/arena.hpp).
struct NoPayload {
  constexpr bool operator==(const NoPayload &) const = default;
};
template <typename Tag> struct Index {
  std::uint32_t value = 0;
  constexpr bool operator==(const Index &) const = default;
};
// A named input's place among its kind, or a seed's direction.
using Slot = Index<struct SlotTag>;
// Where a VecConst's or MatConst's data sits in Model::tensors.
using TensorRef = Index<struct TensorRefTag>;
// How long a node's list is.
using Count = Index<struct CountTag>;
// The element a Gather reads or an AddAt adds to, or the row a RowOf takes.
using Element = Index<struct ElementTag>;
// Whether a MatRow or MatVec reads its matrix as M^T.
enum class Orientation : std::uint32_t { plain, transposed };

// An extern node's payload: which registered function, and which component of
// its answer -- 0 the value, 1..k the first partials, then the upper triangle
// of the second, row-major.  A half-word each; fits() is the one bound.
struct ExternRef {
  std::uint32_t function = 0;
  std::uint32_t component = 0;
  static constexpr std::uint32_t limit = 1u << 16;

  [[nodiscard]] static constexpr ExternRef of(std::uint32_t word) noexcept {
    return {.function = word >> 16, .component = word & (limit - 1)};
  }
  [[nodiscard]] constexpr std::uint32_t packed() const noexcept {
    return (function << 16) | component;
  }
  [[nodiscard]] static constexpr bool fits(std::size_t function,
                                           std::size_t components) noexcept {
    return function < limit && components <= limit;
  }
  constexpr bool operator==(const ExternRef &) const = default;
};

// A Map's or Zip's payload is the OpCode it applies elementwise.
using Payloads = boost::mp11::mp_list<NoPayload, Slot, TensorRef, Count,
                                      Element, Orientation, OpCode, ExternRef>;
template <typename P>
  requires boost::mp11::mp_contains<Payloads, P>::value
inline constexpr auto payload_index =
    static_cast<std::uint8_t>(boost::mp11::mp_find<Payloads, P>::value);

// The word a payload is stored as, and back.
template <typename P>
[[nodiscard]] constexpr std::uint32_t to_word(const P &p) noexcept {
  if constexpr (std::same_as<P, NoPayload>) {
    return 0;
  } else if constexpr (std::same_as<P, ExternRef>) {
    return p.packed();
  } else if constexpr (std::is_enum_v<P>) {
    return std::to_underlying(p);
  } else {
    return p.value;
  }
}
template <typename P>
[[nodiscard]] constexpr P from_word(std::uint32_t word) noexcept {
  if constexpr (std::same_as<P, NoPayload>) {
    return {};
  } else if constexpr (std::same_as<P, ExternRef>) {
    return ExternRef::of(word);
  } else if constexpr (std::is_enum_v<P>) {
    return static_cast<P>(word);
  } else {
    return {word};
  }
}

// What a node's value is: one scalar, a row of scalars per point, or a table
// a MatRow reads and nothing sweeps.
enum class Yields : std::uint8_t { Scalar, Row, Table };

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
  std::uint8_t payload = payload_index<NoPayload>; // an index into Payloads
  Yields yields = Yields::Scalar;
};

// Arity and kind are which sub-table a row sits in, so the groups fill one
// array in one pass each.  Each row lands at its own enumerator, so nothing
// depends on the table order matching the enum's.
inline constexpr std::array op_info = [] {
  std::array<OpInfo, op_count> t{};
#define DDX_RT_ROW(fn, Op, label, role, ...)                                   \
  t[static_cast<std::size_t>(OpCode::Op)] = {                                  \
      #fn, label, 0, Role::role == Role::Literal ? Kind::Const : Kind::Input,  \
      Role::role == Role::Literal ? payload_index<NoPayload>                   \
                                  : payload_index<Slot>};
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
#define DDX_RT_ROW(fn, Op, label, functor, yields)                             \
  t[static_cast<std::size_t>(OpCode::Op)] = {                                  \
      #fn, label, 3, Kind::Ternary, payload_index<NoPayload>, Yields::yields};
  DDX_RT_TERNARY_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, arity, kind)                                 \
  t[static_cast<std::size_t>(OpCode::Op)] = {#fn, label, arity, Kind::kind,    \
                                             payload_index<ExternRef>};
  DDX_RT_EXTERN_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, arity, kind, binary)                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {#fn, label, arity, Kind::kind,    \
                                             payload_index<Count>};
  DDX_RT_NARY_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, arity, kind, payload, yields)                \
  t[static_cast<std::size_t>(OpCode::Op)] = {                                  \
      #fn, label, arity, Kind::kind, payload_index<payload>, Yields::yields};
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

// An OpCode names a row: a byte from outside -- a Python int -- is
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
[[nodiscard]] constexpr Yields yields_of(OpCode op) noexcept {
  return detail::info(op).yields;
}
// The type of what an `Op` node carries: what make() takes and get() hands
// back for it.
template <OpCode Op>
using payload_of_t = boost::mp11::mp_at_c<Payloads, detail::info(Op).payload>;
// An op as a value whose type names it: what lets make() check an arity and a
// payload type as it compiles.
template <OpCode Op>
inline constexpr std::integral_constant<OpCode, Op> opcode{};

// By label, at compile time for the bridge and at run time for the text
// parser.
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
// The binary op a Nary node folds its terms with.
[[nodiscard]] constexpr OpCode fold_op(OpCode op) noexcept {
  switch (op) {
#define DDX_RT_FOLD(fn, Op, label, arity, kind, binary)                        \
  case OpCode::Op:                                                             \
    return OpCode::binary;
    DDX_RT_NARY_TABLE(DDX_RT_FOLD)
#undef DDX_RT_FOLD
  default:
    std::unreachable();
  }
}
// The Nary node that folds with `binary`, if one does.
[[nodiscard]] constexpr std::optional<OpCode> fold_of(OpCode binary) noexcept {
#define DDX_RT_FOLD(fn, Op, label, arity, kind, bin)                           \
  if (binary == OpCode::bin) {                                                 \
    return OpCode::Op;                                                         \
  }
  DDX_RT_NARY_TABLE(DDX_RT_FOLD)
#undef DDX_RT_FOLD
  return std::nullopt;
}
// Whether `a` is where the node's list starts in Model::lists rather than an
// operand: an extern's arguments, a Pack's or StackRows' items, a sum's terms
// -- every node whose payload says how long the list is.
[[nodiscard]] constexpr bool has_list(OpCode op) noexcept {
  const auto payload = detail::info(op).payload;
  return payload == payload_index<Count> || payload == payload_index<ExternRef>;
}
// What text and Python may call by name: a scalar operation on operands, not
// a leaf, not a function the program registered, not a list and not a tensor
// row.
[[nodiscard]] constexpr bool callable(OpCode op) noexcept {
  return !is_leaf(op) && !has_list(op) && kind_of(op) != Kind::Tensor &&
         yields_of(op) == Yields::Scalar;
}
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

// How many input columns of each kind a call passes, in the order it passes
// them: the named kinds in table order, then a seeded product's direction.
struct Inputs {
  std::array<std::size_t, named_count> by_kind{};
  std::size_t directions = 0;

  [[nodiscard]] constexpr std::size_t count(OpCode op) const noexcept {
    return by_kind[*named_index(op)];
  }
  [[nodiscard]] constexpr std::size_t named() const noexcept {
    return std::ranges::fold_left(by_kind, 0uz, std::plus<>{});
  }
  [[nodiscard]] constexpr std::size_t total() const noexcept {
    return named() + directions;
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
