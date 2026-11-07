#pragma once

// Factories and reductions for vector and matrix expressions. The % operator
// is dot() for two vectors and gemv() for a matrix and a vector.

#include "rt/arena.hpp"
#include "rt/expressions.hpp"
#include "util/error.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>

#include <cstdint>
#include <format>
#include <initializer_list>
#include <limits>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ddx::rt {

namespace detail {
// The arena of the first scalar that has one, else the callback's arena.
template <impl::Numeric T>
[[nodiscard]] Arena<T> *arena_of(std::span<const Expression<T>> scalars) {
  Arena<T> *const b = Router::first_arena(scalars);
  return b != nullptr ? b : current_arena<T>;
}

// A poisoned result for a refused factory call.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> refused(errc why) noexcept {
  return Router::poison<T>(why);
}
// Runs make in the callback's arena, or refuses outside an equation callback.
template <impl::Numeric T> [[nodiscard]] Expression<T> in_arena(auto &&make) {
  return current_arena<T> == nullptr ? refused<T>(errc::outside_equation)
                                     : DDX_FWD(make)(*current_arena<T>);
}
// The poison of the first poisoned operand, if any.
[[nodiscard]] constexpr std::optional<errc> poisoned(const auto &...es) {
  std::optional<errc> why;
  ((why = why ? why : Router::why(es)), ...);
  return why;
}
// The shape of a node, including a table, which shape_of() does not cover.
template <impl::Numeric T>
[[nodiscard]] constexpr Shape matrix_shape(const Arena<T> &b,
                                           const Expression<T> &m) {
  if (b.op_of(m.id()) != OpCode::MatConst) {
    return m.shape();
  }
  const auto &data = b.tensors()[get<TensorRef>(b[m.id()]).value];
  return {.rows = data.rows, .cols = data.cols};
}
// The error if any operand is not a vector node, such as a pending literal or
// a scalar node.
[[nodiscard]] constexpr std::optional<errc> not_a_vector(const auto &...vs) {
  if (const auto why = poisoned(vs...)) {
    return why;
  }
  return ((vs.size() == 0) || ...) ? std::optional{errc::tensor_shape}
                                   : std::nullopt;
}
} // namespace detail

// Packs scalars into a vector, one element each, so a vector of n symbols is
// still n Var leaves.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T>
pack(Arena<T> &b, std::span<const Expression<T>> scalars) {
  if (scalars.empty()) {
    return detail::refused<T>(errc::tensor_shape);
  }
  for (const auto &s : scalars) {
    if (const auto why = detail::Router::why(s)) {
      return detail::refused<T>(*why);
    }
    if (!detail::Router::scalar(s)) {
      return detail::refused<T>(errc::tensor_shape);
    }
  }
  const auto ids =
      scalars | std::views::transform([&b](const auto &s) { return s.id(b); }) |
      impl::to<boost::container::small_vector<NodeId, 8>>();
  return {b, b.pack(ids)};
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> pack(std::span<const Expression<T>> scalars) {
  Arena<T> *const b = detail::arena_of(scalars);
  return b == nullptr ? detail::refused<T>(errc::outside_equation)
                      : pack(*b, scalars);
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> pack(std::initializer_list<Expression<T>> scalars) {
  return pack(std::span<const Expression<T>>{scalars});
}
template <impl::Numeric T>
[[nodiscard]] Expression<T> pack(const std::vector<Expression<T>> &scalars) {
  return pack(std::span<const Expression<T>>{scalars});
}

namespace detail {
// Packs n inputs of the given kind named name[0] through name[n-1].
template <impl::Numeric T>
[[nodiscard]] Expression<T> indexed(Arena<T> &b, OpCode kind,
                                    std::string_view name, std::size_t n) {
  const auto leaves = std::views::iota(0uz, n) |
                      std::views::transform([&](std::size_t i) {
                        return named(b, kind, std::format("{}[{}]", name, i));
                      }) |
                      impl::to<std::vector<Expression<T>>>();
  return pack(b, std::span<const Expression<T>>{leaves});
}
} // namespace detail

// A vector of n symbols named name[0] through name[n-1]; vec_param makes
// parameters, which are never differentiated.
template <impl::Numeric T>
[[nodiscard]] Expression<T> vec(Arena<T> &b, std::string_view name,
                                std::size_t n) {
  return detail::indexed(b, OpCode::Var, name, n);
}
template <impl::Numeric T>
[[nodiscard]] Expression<T> vec_param(Arena<T> &b, std::string_view name,
                                      std::size_t n) {
  return detail::indexed(b, OpCode::Param, name, n);
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> vec(std::string_view name, std::size_t n) {
  return detail::in_arena<T>([&](Arena<T> &b) { return vec(b, name, n); });
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> vec_param(std::string_view name, std::size_t n) {
  return detail::in_arena<T>(
      [&](Arena<T> &b) { return vec_param(b, name, n); });
}

// A constant vector or a column-major matrix.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> vec_constant(Arena<T> &b,
                                                   std::span<const T> data) {
  if (data.empty()) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return {b, b.tensor_const(static_cast<std::uint32_t>(data.size()), 1, data)};
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T>
vec_constant(std::span<const std::type_identity_t<T>> data) {
  return detail::in_arena<T>(
      [&](Arena<T> &b) { return vec_constant(b, data); });
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> vec_constant(std::initializer_list<T> data) {
  return vec_constant(std::span<const T>{data});
}

template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T>
mat_constant(Arena<T> &b, std::size_t rows, std::size_t cols,
             std::span<const T> data) {
  // Both dimensions are checked against 32 bits first, so rows * cols cannot
  // wrap.
  constexpr std::size_t most = std::numeric_limits<std::uint32_t>::max();
  if (rows == 0 || cols == 0 || rows > most || cols > most ||
      data.size() != rows * cols) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return {b, b.tensor_const(static_cast<std::uint32_t>(rows),
                            static_cast<std::uint32_t>(cols), data)};
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T>
mat_constant(std::size_t rows, std::size_t cols,
             std::span<const std::type_identity_t<T>> data) {
  return detail::in_arena<T>(
      [&](Arena<T> &b) { return mat_constant(b, rows, cols, data); });
}

// Reductions to a scalar.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> sum(const Expression<T> &v) {
  if (const auto why = detail::not_a_vector(v)) {
    return detail::refused<T>(*why);
  }
  Arena<T> &b = *detail::Router::arena(v);
  return {b, b.make(OpCode::Sum, v.id())};
}
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> dot(const Expression<T> &u,
                                          const Expression<T> &v) {
  if (const auto why = detail::poisoned(u, v)) {
    return detail::refused<T>(*why);
  }
  // Either side may be a literal, which takes the other side's width.
  Arena<T> *const b = detail::Router::arena(u) ? detail::Router::arena(u)
                                               : detail::Router::arena(v);
  const Shape shape = u.size() ? u.shape() : v.shape();
  if (b == nullptr || shape.size() == 0 ||
      (!u.pending() && !v.pending() &&
       detail::Router::arena(u) != detail::Router::arena(v)) ||
      (u.size() != 0 && v.size() != 0 && u.shape() != v.shape()) ||
      (!u.pending() && u.size() == 0) || (!v.pending() && v.size() == 0)) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return {*b, b->make(opcode<OpCode::Dot>,
                      {.left = u.id(*b, shape), .right = v.id(*b, shape)})};
}

// M v, or M^T v when transposed, for a constant table or a computed matrix.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T>
gemv(const Expression<T> &m, const Expression<T> &v, Orientation orientation) {
  const bool transposed = orientation == Orientation::transposed;
  if (const auto why = detail::poisoned(m, v)) {
    return detail::refused<T>(*why);
  }
  Arena<T> *const b = detail::Router::arena(m);
  if (b == nullptr || v.size() == 0 || detail::Router::arena(v) != b) {
    return detail::refused<T>(errc::tensor_shape);
  }
  const Shape shape = detail::matrix_shape(*b, m);
  if (shape.cols == 1 || v.size() != (transposed ? shape.rows : shape.cols)) {
    return detail::refused<T>(errc::tensor_shape);
  }
  const NodeId rows =
      b->op_of(m.id()) == OpCode::MatConst
          ? b->make(opcode<OpCode::MatRow>, Orientation::plain, m.id())
          : m.id();
  return {*b, b->make(opcode<OpCode::MatVec>, orientation,
                      {.matrix = rows, .vec = v.id()})};
}
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T>
gemv(const Expression<T> &m, const Expression<T> &v, bool transposed = false) {
  return gemv(m, v, transposed ? Orientation::transposed : Orientation::plain);
}

namespace detail {
// A transposed matrix usable only on the left of %, where it builds
// gemv(M, v, true) rather than a node of its own.
template <impl::Numeric T> struct Transposed {
  Expression<T> m;

  [[nodiscard]] friend constexpr Expression<T>
  operator%(const Transposed &t, const Expression<T> &v) {
    return gemv(t.m, v, Orientation::transposed);
  }
};
} // namespace detail

template <impl::Numeric T>
[[nodiscard]] constexpr detail::Transposed<T>
transpose(const Expression<T> &m) {
  return {m};
}

// M % v is gemv(M, v) and u % v is dot(u, v); a matrix on the right is
// refused.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> operator%(const Expression<T> &l,
                                                const Expression<T> &r) {
  const auto is_matrix = [](const Expression<T> &e) {
    Arena<T> *const b = detail::Router::arena(e);
    return b != nullptr && !detail::Router::why(e) &&
           detail::matrix_shape(*b, e).cols > 1;
  };
  if (is_matrix(r) && !detail::Router::why(l)) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return is_matrix(l) ? gemv(l, r) : dot(l, r);
}

// The outer product u w^T.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> outer(const Expression<T> &u,
                                            const Expression<T> &w) {
  if (const auto why = detail::not_a_vector(u, w)) {
    return detail::refused<T>(*why);
  }
  Arena<T> *const b = detail::Router::arena(u);
  if (detail::Router::arena(w) != b || u.shape().cols != 1 ||
      w.shape().cols != 1) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return {*b,
          b->make(opcode<OpCode::Outer>, {.left = u.id(), .right = w.id()})};
}

namespace detail {
// v + s e_i, which adds a scalar to element i; this is the adjoint of v[i].
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T>
add_at(const Expression<T> &v, const Expression<T> &s, std::size_t i) {
  if (const auto why = detail::not_a_vector(v)) {
    return detail::refused<T>(*why);
  }
  if (const auto why = detail::Router::why(s)) {
    return detail::refused<T>(*why);
  }
  if (i >= v.size() || !detail::Router::scalar(s)) {
    return detail::refused<T>(errc::tensor_shape);
  }
  Arena<T> &b = *detail::Router::arena(v);
  return {b,
          b.make(opcode<OpCode::AddAt>, Element{static_cast<std::uint32_t>(i)},
                 {.vec = v.id(), .scalar = s.id(b)})};
}
} // namespace detail

} // namespace ddx::rt
