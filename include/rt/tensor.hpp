#pragma once

// Vectors in a runtime graph: the factories and reductions over a handle that
// is itself vector-valued (RTExpression::size() > 0).
//
//   auto x = rt::vec("x", n);                 // n symbols x[0] .. x[n-1],
//   packed auto C = rt::mat_const(n, n, data);       // column-major auto g =
//   rt::dot(x, rt::gemv(C, x)) + rt::sum(log(x) * x);
//
// The arithmetic is RTExpression's own: elementwise over a vector, a Scale
// for a scalar times one, and a literal takes the width of the vector it
// meets.  A matrix is a constant table that only gemv() reads.

#include "rt/builder.hpp"
#include "rt/expressions.hpp"
#include "util/error.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>

#include <cstdint>
#include <format>
#include <initializer_list>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ddx::rt {

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

// The handle over a tensor node, or the refusal the factories share.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> tensor(Builder<T> &b, NodeId id) {
  return {b, id};
}
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> refused(errc why) noexcept {
  return Rooting::poison<T>(why);
}
// Whether `v` is a vector node of this arena, poisoned, or neither: a pending
// literal has no width and a scalar node is not a vector.
template <impl::Numeric T>
[[nodiscard]] constexpr std::optional<errc>
not_a_vector(const RTExpression<T> &v) {
  if (const auto why = Rooting::why(v)) {
    return why;
  }
  return v.size() == 0 ? std::optional{errc::tensor_shape} : std::nullopt;
}
} // namespace detail

// The vector of `scalars`, in order.  Every scalar is one element, so a
// vector of n symbols is n Var leaves and symbols() is unchanged.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T>
pack(Builder<T> &b, std::span<const RTExpression<T>> scalars) {
  if (scalars.empty()) {
    return detail::refused<T>(errc::tensor_shape);
  }
  for (const auto &s : scalars) {
    if (const auto why = detail::Rooting::why(s)) {
      return detail::refused<T>(*why);
    }
    if (s.size() != 0) {
      return detail::refused<T>(errc::tensor_shape);
    }
  }
  const auto ids =
      scalars | std::views::transform([&b](const auto &s) { return s.id(b); }) |
      impl::to<boost::container::small_vector<NodeId, 8>>();
  return detail::tensor(b, b.pack(ids));
}
template <impl::Numeric T = double>
[[nodiscard]] RTExpression<T> pack(std::span<const RTExpression<T>> scalars) {
  Builder<T> *const b = detail::arena_of(scalars);
  return b == nullptr ? detail::refused<T>(errc::no_arena) : pack(*b, scalars);
}
template <impl::Numeric T = double>
[[nodiscard]] RTExpression<T>
pack(std::initializer_list<RTExpression<T>> scalars) {
  return pack(std::span<const RTExpression<T>>{scalars});
}
template <impl::Numeric T>
[[nodiscard]] RTExpression<T>
pack(const std::vector<RTExpression<T>> &scalars) {
  return pack(std::span<const RTExpression<T>>{scalars});
}

// n symbols named `name[0]` .. `name[n-1]`, packed: the one piece of sugar.
template <impl::Numeric T>
[[nodiscard]] RTExpression<T> vec(Builder<T> &b, std::string_view name,
                                  std::size_t n) {
  const auto symbols = std::views::iota(0uz, n) |
                       std::views::transform([&](std::size_t i) {
                         return var(b, std::format("{}[{}]", name, i));
                       }) |
                       impl::to<std::vector<RTExpression<T>>>();
  return pack(b, std::span<const RTExpression<T>>{symbols});
}
template <impl::Numeric T = double>
[[nodiscard]] RTExpression<T> vec(std::string_view name, std::size_t n) {
  return detail::current_arena<T> == nullptr
             ? detail::refused<T>(errc::no_arena)
             : vec(*detail::current_arena<T>, name, n);
}

// A constant vector or matrix, column-major.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> vec_const(Builder<T> &b,
                                                  std::span<const T> data) {
  if (data.empty()) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return detail::tensor(
      b, b.tensor_const(static_cast<std::uint32_t>(data.size()), 1, data));
}
template <impl::Numeric T = double>
[[nodiscard]] RTExpression<T>
vec_const(std::span<const std::type_identity_t<T>> data) {
  return detail::current_arena<T> == nullptr
             ? detail::refused<T>(errc::no_arena)
             : vec_const(*detail::current_arena<T>, data);
}
template <impl::Numeric T = double>
[[nodiscard]] RTExpression<T> vec_const(std::initializer_list<T> data) {
  return vec_const(std::span<const T>{data});
}

template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T>
mat_const(Builder<T> &b, std::size_t rows, std::size_t cols,
          std::span<const T> data) {
  if (rows == 0 || cols == 0 || data.size() != rows * cols) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return detail::tensor(b,
                        b.tensor_const(static_cast<std::uint32_t>(rows),
                                       static_cast<std::uint32_t>(cols), data));
}
template <impl::Numeric T = double>
[[nodiscard]] RTExpression<T>
mat_const(std::size_t rows, std::size_t cols,
          std::span<const std::type_identity_t<T>> data) {
  return detail::current_arena<T> == nullptr
             ? detail::refused<T>(errc::no_arena)
             : mat_const(*detail::current_arena<T>, rows, cols, data);
}

// The reductions to a scalar.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> sum(const RTExpression<T> &v) {
  if (const auto why = detail::not_a_vector(v)) {
    return detail::refused<T>(*why);
  }
  Builder<T> &b = *detail::Rooting::builder(v);
  return {b, b.make(OpCode::Sum, v.id())};
}
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> dot(const RTExpression<T> &u,
                                            const RTExpression<T> &v) {
  for (const auto *e : {&u, &v}) {
    if (const auto why = detail::Rooting::why(*e)) {
      return detail::refused<T>(*why);
    }
  }
  // Either side may be a literal, which takes the other's width.
  Builder<T> *const b = detail::Rooting::builder(u)
                            ? detail::Rooting::builder(u)
                            : detail::Rooting::builder(v);
  const std::uint32_t w = std::max(u.size(), v.size());
  if (b == nullptr || w == 0 ||
      (u.size() != 0 && v.size() != 0 && u.size() != v.size()) ||
      (!u.pending() && u.size() == 0) || (!v.pending() && v.size() == 0)) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return {*b, b->make(OpCode::Dot, u.id(*b, w), v.id(*b, w))};
}

// M v, or M^T v.  The matrix is a constant, so neither differentiates
// through it.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> gemv(const RTExpression<T> &m,
                                             const RTExpression<T> &v,
                                             bool transposed = false) {
  for (const auto *e : {&m, &v}) {
    if (const auto why = detail::Rooting::why(*e)) {
      return detail::refused<T>(*why);
    }
  }
  Builder<T> *const b = detail::Rooting::builder(m);
  if (b == nullptr || b->op_of(m.id()) != OpCode::MatConst || v.size() == 0 ||
      detail::Rooting::builder(v) != b) {
    return detail::refused<T>(errc::tensor_shape);
  }
  const auto &table = b->tensors()[(*b)[m.id()].slot];
  if (v.size() != (transposed ? table.rows : table.cols)) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return {*b, b->make(OpCode::Gemv, m.id(), v.id(), no_node, transposed)};
}

namespace detail {
// v + s e_i: the vector with one element moved by a scalar -- the adjoint of
// v[i].
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T>
add_at(const RTExpression<T> &v, const RTExpression<T> &s, std::size_t i) {
  if (const auto why = detail::not_a_vector(v)) {
    return detail::refused<T>(*why);
  }
  if (const auto why = detail::Rooting::why(s)) {
    return detail::refused<T>(*why);
  }
  if (i >= v.size() || s.size() != 0) {
    return detail::refused<T>(errc::tensor_shape);
  }
  Builder<T> &b = *detail::Rooting::builder(v);
  return {b, b.make(OpCode::AddAt, v.id(), s.id(b), no_node,
                    static_cast<std::uint32_t>(i))};
}
} // namespace detail

} // namespace ddx::rt
