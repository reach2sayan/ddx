#pragma once

#include "ops/unary_math.hpp" // DDX_UNARY_MATH_TABLE
#include "util/meta.hpp"
#include "util/ranges.hpp"

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

// The runtime mirror of the compile-time operation set. Each row starts with
// (factory spelling, enumerator, label), as DDX_UNARY_MATH_TABLE does.
namespace ddx::rt {

// Leaves; every one but Const reads an input column, and only Var is
// differentiated. Named kinds are listed in the order a point passes them.
#define DDX_RT_NAMED_TABLE(X)                                                  \
  X(var, Var, "var", Slot)                                                     \
  X(param, Param, "param", Slot)                                               \
  X(implicit, Implicit, "implicit", Slot)
#define DDX_RT_LEAF_TABLE(X)                                                   \
  X(constant, Const, "const", NoPayload)                                       \
  DDX_RT_NAMED_TABLE(X)                                                        \
  X(seed, Seed, "seed", Slot)

// The last cell names the ops/adjoints.hpp descriptor holding the reverse
// rule, shared with the compile-time ops.
#define DDX_RT_UNARY_TABLE(X)                                                  \
  X(neg, Neg, "-", std::negate<>, NegateOpFn)                                  \
  X(abs, Abs, "abs", impl::detail::abs_impl, AbsOpFn)                          \
  X(sign, Sign, "sign", impl::detail::sign_impl, SignOpFn)

#define DDX_RT_BINARY_TABLE(X)                                                 \
  X(add, Add, "+", std::plus<>, SumOpFn)                                       \
  X(mul, Mul, "*", std::multiplies<>, MultiplyOpFn)                            \
  X(div, Div, "/", std::divides<>, DivideOpFn)                                 \
  DDX_BINARY_MATH_TABLE(X)

// Comparisons need no adjoint descriptor, as their derivative is zero.
#define DDX_RT_COMPARE_TABLE(X)                                                \
  X(lt, Lt, "<", impl::detail::lt_impl)                                        \
  X(le, Le, "<=", impl::detail::le_impl)

// c != 0 ? t : f with both arms evaluated; VSelect is the tensor form. The
// last cells are the yield and the operand names.
#define DDX_RT_TERNARY_TABLE(X)                                                \
  X(select, Select, "select", impl::detail::select_impl, Scalar,               \
    (cond, then, otherwise))                                                   \
  X(vselect, VSelect, "vselect", impl::detail::select_impl, Row,               \
    (cond, then, otherwise))

// A call to a user function over a list of arguments. Its ExternRef payload
// picks the function and the component: the value or one partial.
#define DDX_RT_EXTERN_TABLE(X) X(extern_call, Extern, "extern", 0, Extern)
// A left fold ((t0 op t1) op t2) ... over a list, built by chains of +=, *=
// or /=. The last cell is the binary op; sums fuse product terms into fmas.
#define DDX_RT_NARY_TABLE(X)                                                   \
  X(sum_n, SumN, "sum_n", 0, Nary, Add)                                        \
  X(prod_n, ProdN, "prod_n", 0, Nary, Mul)                                     \
  X(quot_n, QuotN, "quot_n", 0, Nary, Div)

// Tensor nodes yield a column-major row per point, or a scalar read off one.
// The cells after the label are arity, Kind, payload, yield and operand names.
#define DDX_RT_TENSOR_TABLE(X)                                                 \
  X(vec_const, VecConst, "vconst", 0, Tensor, TensorRef, Row, ())              \
  X(mat_const, MatConst, "mconst", 0, Tensor, TensorRef, Table, ())            \
  X(pack, Pack, "pack", 0, Tensor, Count, Row, ())                             \
  X(gather, Gather, "gather", 1, Tensor, Element, Scalar, (vec))               \
  X(vmap, Map, "vmap", 1, Tensor, OpCode, Row, (vec))                          \
  X(vzip, Zip, "vzip", 2, Tensor, OpCode, Row, (left, right))                  \
  X(vscale, Scale, "vscale", 2, Tensor, NoPayload, Row, (scalar, vec))         \
  X(vsum, Sum, "vsum", 1, Tensor, NoPayload, Scalar, (vec))                    \
  X(vdot, Dot, "vdot", 2, Tensor, NoPayload, Scalar, (left, right))            \
  X(add_at, AddAt, "add_at", 2, Tensor, Element, Row, (vec, scalar))           \
  X(mat_row, MatRow, "mrow", 1, Tensor, Orientation, Row, (source))            \
  X(diag, Diag, "diag", 1, Tensor, NoPayload, Row, (vec))                      \
  X(outer, Outer, "outer", 2, Tensor, NoPayload, Row, (left, right))           \
  X(row_scale, RowScale, "row_scale", 2, Tensor, NoPayload, Row,               \
    (scale, matrix))                                                           \
  X(col_scale, ColScale, "col_scale", 2, Tensor, NoPayload, Row,               \
    (matrix, scale))                                                           \
  X(matvec, MatVec, "matvec", 2, Tensor, Orientation, Row, (matrix, vec))      \
  X(matmul, MatMul, "matmul", 2, Tensor, NoPayload, Row, (left, right))        \
  X(stack_rows, StackRows, "stack_rows", 0, Tensor, Count, Row, ())            \
  X(row_of, RowOf, "row_of", 1, Tensor, Element, Row, (matrix))

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

// Typed payloads: each opcode row names the type its node's payload word is
// read back as.
struct NoPayload {
  constexpr bool operator==(const NoPayload &) const = default;
};
template <typename Tag> struct Index {
  std::uint32_t value = 0;
  constexpr bool operator==(const Index &) const = default;
};
// A named input's index within its kind, or a seed's direction.
using Slot = Index<struct SlotTag>;
// The index of a VecConst's or MatConst's data in Model::tensors.
using TensorRef = Index<struct TensorRefTag>;
// The length of a node's operand list.
using Count = Index<struct CountTag>;
// The element a Gather reads or an AddAt adds to, or the row RowOf takes.
using Element = Index<struct ElementTag>;
// Whether a MatRow or MatVec reads its matrix as M^T.
enum class Orientation : std::uint32_t { plain, transposed };

// The function index and answer component, 16 bits each. Component 0 is the
// value, then the first partials, then the second's row-major upper triangle.
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

// Every payload type. A Map's or Zip's payload is its elementwise OpCode.
using Payloads = meta::list<NoPayload, Slot, TensorRef, Count, Element,
                            Orientation, OpCode, ExternRef>;
template <typename P>
  requires(meta::contains(Payloads{}, meta::type_c<P>))
inline constexpr auto payload_index =
    static_cast<std::uint8_t>(meta::index_of(Payloads{}, meta::type_c<P>));

// Converts a payload to its stored word and back.
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

// A node's value: a scalar, a row of scalars per point, or a table that only
// a MatRow reads.
enum class Yields : std::uint8_t { Scalar, Row, Table };

// The node classification that every sweep, emitter and checker switches on.
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

struct OpInfo {
  std::string_view name; // the factory spelling and the Python name
  std::string_view label;
  std::uint8_t arity;
  Kind kind;
  std::uint8_t payload = payload_index<NoPayload>; // an index into Payloads
  Yields yields = Yields::Scalar;
};

// Fills one entry per opcode from each sub-table, indexed by enumerator so
// the table order does not matter.
inline constexpr std::array op_info = [] {
  std::array<OpInfo, op_count> t{};
#define DDX_RT_ROW(fn, Op, label, payload)                                     \
  t[static_cast<std::size_t>(OpCode::Op)] = {                                  \
      #fn, label, 0,                                                           \
      std::same_as<payload, NoPayload> ? Kind::Const : Kind::Input,            \
      payload_index<payload>};
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
#define DDX_RT_ROW(fn, Op, label, functor, yields, names)                      \
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
#define DDX_RT_ROW(fn, Op, label, arity, kind, payload, yields, names)         \
  t[static_cast<std::size_t>(OpCode::Op)] = {                                  \
      #fn, label, arity, Kind::kind, payload_index<payload>, Yields::yields};
  DDX_RT_TENSOR_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
  return t;
}();

// Labels must be unique, since opcode_of() and the compile-time ops look up
// enumerators by label.
static_assert([] {
  auto labels = op_info;
  std::ranges::sort(labels, {}, &OpInfo::label);
  return std::ranges::adjacent_find(labels, {}, &OpInfo::label) == labels.end();
}());

// Not range-checked; external bytes such as Python ints are checked where
// they are cast to OpCode.
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
// Whether an op's operands are passed by name, for tensor ops and VSelect
// where swapped operands would still type-check. Scalar ops stay positional.
[[nodiscard]] constexpr bool named_operands(OpCode op) noexcept {
  return arity_of(op) >= 2 &&
         (kind_of(op) == Kind::Tensor || yields_of(op) == Yields::Row);
}
// The payload type of an Op node, as make() takes and get() returns it.
template <OpCode Op>
using payload_of_t = DDX_TYPE(meta::at<detail::info(Op).payload>(Payloads{}));
// An opcode as a type, so make() can check arity and payload at compile time.
template <OpCode Op>
inline constexpr std::integral_constant<OpCode, Op> opcode{};

// Looks up an opcode by label, for the bridge and the text parser.
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

// Leaves come first in the op table, so this is one compare instead of a
// table load on every sweep step.
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
// The Nary opcode that folds with the given binary op, if any.
[[nodiscard]] constexpr std::optional<OpCode> fold_of(OpCode binary) noexcept {
#define DDX_RT_FOLD(fn, Op, label, arity, kind, bin)                           \
  if (binary == OpCode::bin) {                                                 \
    return OpCode::Op;                                                         \
  }
  DDX_RT_NARY_TABLE(DDX_RT_FOLD)
#undef DDX_RT_FOLD
  return std::nullopt;
}
// Whether the node reads a list in Model::lists rather than fixed operands.
[[nodiscard]] constexpr bool has_list(OpCode op) noexcept {
  const auto payload = detail::info(op).payload;
  return payload == payload_index<Count> || payload == payload_index<ExternRef>;
}
// Whether text and Python may call the op by name: scalar operand ops only.
[[nodiscard]] constexpr bool callable(OpCode op) noexcept {
  return !is_leaf(op) && !has_list(op) && kind_of(op) != Kind::Tensor &&
         yields_of(op) == Yields::Scalar;
}
// The index of a leaf's named kind in table order, if it has one.
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

// One T per named kind. GCC before 15 double-frees when constant evaluation
// copies an array of allocating elements, so there it copies element-wise.
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
  // At run time a plain copy; during constant evaluation an empty array that
  // the constructor body fills element by element.
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

// Input column counts per kind, in call order: the named kinds in table order,
// then the seeded direction.
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

// The input column a leaf reads. The interpreter, codegen and the ABI all use
// this, so they agree.
[[nodiscard]] constexpr std::size_t input_column(const Inputs &in, OpCode op,
                                                 Slot slot) noexcept {
  // A fixed-length scan, since a variable-length prefix fold vectorises and
  // slows the sweep's leaf step.
  std::array<std::size_t, named_count + 1> start{};
  std::inclusive_scan(in.by_kind.begin(), in.by_kind.end(), start.begin() + 1);
  return start[named_index(op).value_or(named_count)] + slot.value;
}

} // namespace ddx::rt
