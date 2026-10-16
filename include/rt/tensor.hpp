#pragma once

// Vectors in a runtime graph: a handle onto one tensor-valued node, with the
// arithmetic surface that builds more, beside RTExpression's for scalars.
//
//   auto x = rt::vec("x", n);                 // n symbols x[0] .. x[n-1], packed
//   auto C = rt::mat_const(n, n, data);       // column-major
//   auto g = rt::dot(x, rt::gemv(C, x)) + rt::sum(log(x) * x);
//
// RTVector is itself Numeric -- a literal is pending until it meets a vector
// and becomes a VecConst of that width -- so every derivative rule in
// ops/adjoints.hpp instantiates over it elementwise, and no rule is written
// twice.  Comparisons and select are scalar only; a vector has neither.

#include "rt/builder.hpp"
#include "rt/expressions.hpp"
#include "util/error.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ddx::rt {

template <impl::Numeric T = double> class RTMatrix;

namespace detail {
// Befriended by RTVector: the free factories' one way to a poisoned vector,
// as Rooting is the equation's way to an expression's arena.
struct Tensoring {
  template <impl::Numeric T>
  [[nodiscard]] static constexpr RTVector<T> poison(errc why) noexcept;
  template <impl::Numeric T>
  [[nodiscard]] static constexpr RTExpression<T> scalar(errc why) noexcept;
  template <impl::Numeric T>
  [[nodiscard]] static constexpr Builder<T> *
  builder(const RTVector<T> &v) noexcept {
    return v.builder_;
  }
  template <impl::Numeric T>
  [[nodiscard]] static constexpr std::optional<errc>
  why(const RTVector<T> &v) noexcept {
    return v.why_;
  }
};
} // namespace detail

template <impl::Numeric T = double> class RTVector {
  static_assert(std::same_as<T, double>,
                "tensor nodes compute at double: the kernel table's scalar");

public:
  using value_type = T;
  using builder_type = Builder<T>;

  constexpr RTVector() = default;

  // A number alone: the literal every rule manufactures, pending until it
  // meets a vector whose width it takes.  Arithmetic scalars only, so a
  // scalar expression never converts to a vector by accident -- that product
  // is a Scale, spelled below.
  template <impl::CArithmetic V>
  constexpr RTVector(V v) : lit_{static_cast<T>(v)} {}

  constexpr RTVector(Builder<T> &b, NodeId id) noexcept
      : builder_{&b}, id_{id}, width_{width_of(b, id)} {}

  [[nodiscard]] constexpr bool pending() const noexcept {
    return builder_ == nullptr;
  }
  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return why_.has_value();
  }
  // Elements per point; a pending literal has none yet.
  [[nodiscard]] constexpr std::size_t size() const noexcept { return width_; }
  [[nodiscard]] constexpr const T &literal() const noexcept { return lit_; }

  // Give the node an identity in `b`: a pending literal becomes the VecConst
  // of `width` copies.
  [[nodiscard]] constexpr NodeId id(Builder<T> &b, std::uint32_t width) const {
    return builder_ ? id_ : b.filled(width, lit_);
  }
  // The identity it already has.
  [[nodiscard]] constexpr NodeId id() const noexcept { return id_; }

  // One element, as a scalar.
  [[nodiscard]] constexpr RTExpression<T> operator[](std::size_t i) const {
    if (why_) {
      return RTExpression<T>::poison(*why_);
    }
    if (pending()) {
      return RTExpression<T>{lit_};
    }
    if (i >= width_) {
      return RTExpression<T>::poison(errc::tensor_shape);
    }
    return {*builder_, builder_->make_tensor(OpCode::Gather, id_, no_node,
                                             static_cast<std::uint32_t>(i))};
  }

  // Elementwise over one vector: the scalar table's row `op`, as a Map.  Two
  // pending literals fold at T, as RTExpression's do.
  [[nodiscard]] static constexpr RTVector map(OpCode op, const RTVector &u) {
    if (u.why_) {
      return poison(*u.why_);
    }
    if (u.pending()) {
      return RTVector{apply<T>(op, u.lit_)};
    }
    return {*u.builder_, u.builder_->make_tensor(OpCode::Map, u.id_, no_node,
                                                 std::to_underlying(op))};
  }
  [[nodiscard]] static constexpr RTVector zip(OpCode op, const RTVector &l,
                                              const RTVector &r) {
    if (l.why_ || r.why_) {
      return poison(l.why_ ? *l.why_ : *r.why_);
    }
    Builder<T> *const b = l.builder_ ? l.builder_ : r.builder_;
    if (b == nullptr) {
      return RTVector{apply<T>(op, l.lit_, r.lit_)};
    }
    if (l.builder_ && r.builder_ && l.width_ != r.width_) {
      return poison(errc::tensor_shape);
    }
    const std::uint32_t w = l.builder_ ? l.width_ : r.width_;
    return {*b, b->make_tensor(OpCode::Zip, l.id(*b, w), r.id(*b, w),
                               std::to_underlying(op))};
  }
  // s * v for a scalar s: one node, not a broadcast and a Zip.
  [[nodiscard]] static constexpr RTVector scale(const RTExpression<T> &s,
                                                const RTVector &v) {
    if (s.poisoned() || v.why_) {
      return poison(s.poisoned() ? *detail::Rooting::why(s) : *v.why_);
    }
    if (v.pending()) {
      // A scalar node times a literal of no width has no width either.
      return s.pending() ? RTVector{s.literal() * v.lit_}
                         : poison(errc::tensor_shape);
    }
    Builder<T> &b = *v.builder_;
    return {b, b.make_tensor(OpCode::Scale, s.id(b), v.id_)};
  }

  friend constexpr RTVector operator+(const RTVector &l, const RTVector &r) {
    return zip(OpCode::Add, l, r);
  }
  friend constexpr RTVector operator*(const RTVector &l, const RTVector &r) {
    return zip(OpCode::Mul, l, r);
  }
  friend constexpr RTVector operator/(const RTVector &l, const RTVector &r) {
    return zip(OpCode::Div, l, r);
  }
  friend constexpr RTVector operator-(const RTVector &u) {
    return map(OpCode::Neg, u);
  }
  friend constexpr RTVector operator-(const RTVector &l, const RTVector &r) {
    return l + (-r);
  }
  // A scalar -- a number or a scalar expression -- times a vector, either way
  // round, and a vector over one.  Templates, so `2 * v` is this and not an
  // ambiguity between the vector literal and the scalar one; the exclusion
  // comes first so that asking Numeric<RTVector> never asks it again.
  template <typename S>
    requires(!std::same_as<std::remove_cvref_t<S>, RTVector> &&
             impl::Numeric<S>)
  friend constexpr RTVector operator*(const S &s, const RTVector &v) {
    return scale(RTExpression<T>{s}, v);
  }
  template <typename S>
    requires(!std::same_as<std::remove_cvref_t<S>, RTVector> &&
             impl::Numeric<S>)
  friend constexpr RTVector operator*(const RTVector &v, const S &s) {
    return scale(RTExpression<T>{s}, v);
  }
  template <typename S>
    requires(!std::same_as<std::remove_cvref_t<S>, RTVector> &&
             impl::Numeric<S>)
  friend constexpr RTVector operator/(const RTVector &v, const S &s) {
    return scale(RTExpression<T>{1} / RTExpression<T>{s}, v);
  }

  constexpr RTVector &operator+=(const RTVector &o) {
    return *this = *this + o;
  }
  constexpr RTVector &operator-=(const RTVector &o) {
    return *this = *this - o;
  }
  constexpr RTVector &operator*=(const RTVector &o) {
    return *this = *this * o;
  }
  constexpr RTVector &operator/=(const RTVector &o) {
    return *this = *this / o;
  }

  // Every scalar function, elementwise: the same tables RTExpression reads.
#define DDX_RT_VUNFN(fn, Op, label, ...)                                       \
  friend constexpr RTVector fn(const RTVector &u) {                            \
    return map(OpCode::Op, u);                                                 \
  }
  DDX_UNARY_MATH_TABLE(DDX_RT_VUNFN)
  DDX_RT_UNARY_TABLE(DDX_RT_VUNFN)
#undef DDX_RT_VUNFN
#define DDX_RT_VBINFN(fn, Op, label, ...)                                      \
  friend constexpr RTVector fn(const RTVector &l, const RTVector &r) {         \
    return zip(OpCode::Op, l, r);                                              \
  }
  DDX_RT_BINARY_TABLE(DDX_RT_VBINFN)
#undef DDX_RT_VBINFN

private:
  template <impl::Numeric U>
  friend constexpr RTVector<U> pack(Builder<U> &,
                                    std::span<const RTExpression<U>>);
  template <impl::Numeric U>
  friend RTVector<U> pack(std::span<const RTExpression<U>>);
  template <impl::Numeric U>
  friend constexpr RTExpression<U> sum(const RTVector<U> &);
  template <impl::Numeric U>
  friend constexpr RTExpression<U> dot(const RTVector<U> &,
                                       const RTVector<U> &);
  template <impl::Numeric U>
  friend constexpr RTVector<U> gemv(const RTMatrix<U> &, const RTVector<U> &);
  template <impl::Numeric U>
  friend constexpr RTVector<U> gemv_t(const RTMatrix<U> &,
                                      const RTVector<U> &);
  template <impl::Numeric U>
  friend constexpr RTVector<U> add_at(const RTVector<U> &,
                                      const RTExpression<U> &, std::size_t);
  friend struct detail::Rooting;
  friend struct detail::Tensoring;

  [[nodiscard]] static constexpr RTVector poison(errc why) noexcept {
    RTVector v;
    v.why_ = why;
    return v;
  }
  // For the reductions above, which answer a scalar and poison as one.
  [[nodiscard]] static constexpr RTExpression<T> poisoned_scalar(errc why) noexcept {
    return RTExpression<T>::poison(why);
  }

  Builder<T> *builder_ = nullptr;
  NodeId id_ = no_node;
  std::uint32_t width_ = 0;
  T lit_{};
  std::optional<errc> why_;
};

namespace detail {
template <impl::Numeric T>
constexpr RTVector<T> Tensoring::poison(errc why) noexcept {
  return RTVector<T>::poison(why);
}
template <impl::Numeric T>
constexpr RTExpression<T> Tensoring::scalar(errc why) noexcept {
  return RTVector<T>::poisoned_scalar(why);
}
} // namespace detail

// A constant matrix: a handle onto a MatConst, with no algebra of its own
// beyond what gemv() reads.
template <impl::Numeric T> class RTMatrix {
public:
  constexpr RTMatrix() = default;
  constexpr RTMatrix(Builder<T> &b, NodeId id) noexcept
      : builder_{&b}, id_{id}, rows_{b.tensors()[b[id].slot].rows},
        cols_{b.tensors()[b[id].slot].cols} {}
  constexpr explicit RTMatrix(errc why) noexcept : why_{why} {}

  [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return why_.has_value();
  }
  [[nodiscard]] constexpr NodeId id() const noexcept { return id_; }

private:
  template <impl::Numeric U>
  friend constexpr RTVector<U> gemv(const RTMatrix<U> &, const RTVector<U> &);
  template <impl::Numeric U>
  friend constexpr RTVector<U> gemv_t(const RTMatrix<U> &,
                                      const RTVector<U> &);

  Builder<T> *builder_ = nullptr;
  NodeId id_ = no_node;
  std::uint32_t rows_ = 0;
  std::uint32_t cols_ = 0;
  std::optional<errc> why_;
};

namespace detail {
// The arena a vector is built in: the first scalar that names one, else the
// equation() callback's.
template <impl::Numeric T>
[[nodiscard]] Builder<T> *arena_of(std::span<const RTExpression<T>> scalars) {
  for (const auto &s : scalars) {
    if (Builder<T> *const b = Rooting::builder(s)) {
      return b;
    }
  }
  return current_arena<T>;
}
} // namespace detail

// The vector of `scalars`, in order.  Every scalar is one element, so a
// vector of n symbols is n Var leaves and symbols() is unchanged.
template <impl::Numeric T>
[[nodiscard]] constexpr RTVector<T>
pack(Builder<T> &b, std::span<const RTExpression<T>> scalars) {
  if (scalars.empty()) {
    return detail::Tensoring::poison<T>(errc::tensor_shape);
  }
  for (const auto &s : scalars) {
    if (const auto why = detail::Rooting::why(s)) {
      return detail::Tensoring::poison<T>(*why);
    }
  }
  const auto ids = scalars | std::views::transform([&b](const auto &s) {
                     return s.id(b);
                   }) |
                   impl::to<boost::container::small_vector<NodeId, 8>>();
  return {b, b.pack(ids)};
}
template <impl::Numeric T = double>
[[nodiscard]] RTVector<T> pack(std::span<const RTExpression<T>> scalars) {
  Builder<T> *const b = detail::arena_of(scalars);
  return b == nullptr ? detail::Tensoring::poison<T>(errc::no_arena) : pack(*b, scalars);
}
template <impl::Numeric T = double>
[[nodiscard]] RTVector<T> pack(std::initializer_list<RTExpression<T>> scalars) {
  return pack(std::span<const RTExpression<T>>{scalars});
}
template <impl::Numeric T>
[[nodiscard]] RTVector<T> pack(const std::vector<RTExpression<T>> &scalars) {
  return pack(std::span<const RTExpression<T>>{scalars});
}

// n symbols named `name[0]` .. `name[n-1]`, packed: the one piece of sugar.
template <impl::Numeric T>
[[nodiscard]] RTVector<T> vec(Builder<T> &b, std::string_view name,
                              std::size_t n) {
  const auto symbols =
      std::views::iota(0uz, n) | std::views::transform([&](std::size_t i) {
        return var(b, std::format("{}[{}]", name, i));
      }) |
      impl::to<std::vector<RTExpression<T>>>();
  return pack(b, std::span<const RTExpression<T>>{symbols});
}
template <impl::Numeric T = double>
[[nodiscard]] RTVector<T> vec(std::string_view name, std::size_t n) {
  return detail::current_arena<T> == nullptr
             ? detail::Tensoring::poison<T>(errc::no_arena)
             : vec(*detail::current_arena<T>, name, n);
}

// A constant vector or matrix, column-major.
template <impl::Numeric T>
[[nodiscard]] constexpr RTVector<T> vec_const(Builder<T> &b,
                                              std::span<const T> data) {
  if (data.empty()) {
    return detail::Tensoring::poison<T>(errc::tensor_shape);
  }
  return {b, b.tensor_const(static_cast<std::uint32_t>(data.size()), 1, data)};
}
template <impl::Numeric T = double>
[[nodiscard]] RTVector<T> vec_const(std::span<const std::type_identity_t<T>> data) {
  return detail::current_arena<T> == nullptr
             ? detail::Tensoring::poison<T>(errc::no_arena)
             : vec_const(*detail::current_arena<T>, data);
}
template <impl::Numeric T = double>
[[nodiscard]] RTVector<T> vec_const(std::initializer_list<T> data) {
  return vec_const(std::span<const T>{data});
}

template <impl::Numeric T>
[[nodiscard]] constexpr RTMatrix<T> mat_const(Builder<T> &b, std::size_t rows,
                                              std::size_t cols,
                                              std::span<const T> data) {
  if (rows == 0 || cols == 0 || data.size() != rows * cols) {
    return RTMatrix<T>{errc::tensor_shape};
  }
  return {b, b.tensor_const(static_cast<std::uint32_t>(rows),
                            static_cast<std::uint32_t>(cols), data)};
}
template <impl::Numeric T = double>
[[nodiscard]] RTMatrix<T> mat_const(std::size_t rows, std::size_t cols,
                                    std::span<const std::type_identity_t<T>> data) {
  return detail::current_arena<T> == nullptr
             ? RTMatrix<T>{errc::no_arena}
             : mat_const(*detail::current_arena<T>, rows, cols, data);
}

// The reductions to a scalar.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> sum(const RTVector<T> &v) {
  if (v.why_) {
    return detail::Tensoring::scalar<T>(*v.why_);
  }
  if (v.pending()) {
    return detail::Tensoring::scalar<T>(errc::tensor_shape);
  }
  return {*v.builder_, v.builder_->make_tensor(OpCode::Sum, v.id_)};
}
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> dot(const RTVector<T> &u,
                                            const RTVector<T> &v) {
  if (u.why_ || v.why_) {
    return detail::Tensoring::scalar<T>(u.why_ ? *u.why_ : *v.why_);
  }
  Builder<T> *const b = u.builder_ ? u.builder_ : v.builder_;
  if (b == nullptr || (u.builder_ && v.builder_ && u.width_ != v.width_)) {
    return detail::Tensoring::scalar<T>(errc::tensor_shape);
  }
  const std::uint32_t w = u.builder_ ? u.width_ : v.width_;
  return {*b, b->make_tensor(OpCode::Dot, u.id(*b, w), v.id(*b, w))};
}

// M v, and M^T v.  The matrix is a constant, so neither differentiates
// through it.
namespace detail {
template <impl::Numeric T>
[[nodiscard]] constexpr RTVector<T> gemv_impl(const RTMatrix<T> &m,
                                              const RTVector<T> &v,
                                              bool transposed) {
  if (m.poisoned() || v.poisoned()) {
    return detail::Tensoring::poison<T>(errc::tensor_shape);
  }
  const auto in = transposed ? m.rows() : m.cols();
  if (v.pending() || v.size() != in) {
    return detail::Tensoring::poison<T>(errc::tensor_shape);
  }
  Builder<T> &b = *Tensoring::builder(v);
  return {b, b.make_tensor(OpCode::Gemv, m.id(), v.id(), transposed)};
}
} // namespace detail
template <impl::Numeric T>
[[nodiscard]] constexpr RTVector<T> gemv(const RTMatrix<T> &m,
                                         const RTVector<T> &v) {
  return m.poisoned() ? detail::Tensoring::poison<T>(*m.why_)
                      : detail::gemv_impl(m, v, false);
}
template <impl::Numeric T>
[[nodiscard]] constexpr RTVector<T> gemv_t(const RTMatrix<T> &m,
                                           const RTVector<T> &v) {
  return m.poisoned() ? detail::Tensoring::poison<T>(*m.why_)
                      : detail::gemv_impl(m, v, true);
}

// v + s e_i: the vector with one element moved by a scalar.
template <impl::Numeric T>
[[nodiscard]] constexpr RTVector<T> add_at(const RTVector<T> &v,
                                           const RTExpression<T> &s,
                                           std::size_t i) {
  if (v.why_ || s.poisoned()) {
    return detail::Tensoring::poison<T>(v.why_ ? *v.why_ : *detail::Rooting::why(s));
  }
  if (v.pending() || i >= v.width_) {
    return detail::Tensoring::poison<T>(errc::tensor_shape);
  }
  Builder<T> &b = *v.builder_;
  return {b, b.make_tensor(OpCode::AddAt, v.id_, s.id(b),
                           static_cast<std::uint32_t>(i))};
}

} // namespace ddx::rt

namespace ddx::impl {
// Elementwise, so commutative wherever the scalar is.
template <Numeric S>
inline constexpr bool is_commutative_multiply_v<rt::RTVector<S>> =
    is_commutative_multiply_v<S>;
} // namespace ddx::impl
