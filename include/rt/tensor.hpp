#pragma once

// Factories and reductions for vector and matrix expressions. The % operator
// is dot() for two vectors and gemv() for a matrix and a vector.

#include "rt/arena.hpp"
#include "rt/expressions.hpp"
#include "util/error.hpp"
#include "util/meta.hpp"
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
// The shape of a node, including a table, which shape_of() does not cover.
template <impl::Numeric T>
[[nodiscard]] constexpr Shape matrix_shape(const Arena<T> &b,
                                           const Expression<T> &m) {
  if (b.op_of(Router::id(m)) != OpCode::MatConst) {
    return Router::shape(m);
  }
  const auto &data = b.tensors()[get<TensorRef>(b[Router::id(m)]).value];
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

// Packs scalars into a vector, one element each, so a vector of n symbols is
// still n Var leaves.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T>
pack(Arena<T> &b, std::span<const Expression<T>> scalars) {
  if (scalars.empty()) {
    return refused<T>(errc::tensor_shape);
  }
  const auto ids = Router::scalar_ids(b, scalars);
  return ids ? Expression{b, b.pack(*ids)} : refused<T>(ids.error().code);
}

// Packs n inputs of the given kind named name[0] through name[n-1].
template <impl::Numeric T>
[[nodiscard]] Expression<T> indexed(Arena<T> &b, OpCode kind,
                                    std::string_view name, std::size_t n) {
  const auto leaves = std::views::iota(0uz, n) |
                      std::views::transform([&](std::size_t i) {
                        return named(b, kind, std::format("{}[{}]", name, i));
                      }) |
                      impl::to<std::vector>();
  return pack(b, std::span<const Expression<T>>{leaves});
}
template <impl::Numeric T>
[[nodiscard]] Expression<T> vec(Arena<T> &b, std::string_view name,
                                std::size_t n) {
  return indexed(b, OpCode::Var, name, n);
}
template <impl::Numeric T>
[[nodiscard]] Expression<T> vec_param(Arena<T> &b, std::string_view name,
                                      std::size_t n) {
  return indexed(b, OpCode::Param, name, n);
}

template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> vec_constant(Arena<T> &b,
                                                   std::span<const T> data) {
  if (data.empty()) {
    return refused<T>(errc::tensor_shape);
  }
  return {b, b.tensor_const(static_cast<std::uint32_t>(data.size()), 1, data)};
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
    return refused<T>(errc::tensor_shape);
  }
  return {b, b.tensor_const(static_cast<std::uint32_t>(rows),
                            static_cast<std::uint32_t>(cols), data)};
}

// M v, or M^T v when transposed, for a constant table or a computed matrix.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T>
gemv(const Expression<T> &m, const Expression<T> &v, Orientation orientation) {
  const bool transposed = orientation == Orientation::transposed;
  if (const auto why = poisoned(m, v)) {
    return refused<T>(*why);
  }
  Arena<T> *const b = Router::arena(m);
  if (b == nullptr || v.size() == 0 || Router::arena(v) != b) {
    return refused<T>(errc::tensor_shape);
  }
  const Shape shape = matrix_shape(*b, m);
  if (shape.cols == 1 || v.size() != (transposed ? shape.rows : shape.cols)) {
    return refused<T>(errc::tensor_shape);
  }
  const NodeId rows =
      b->op_of(Router::id(m)) == OpCode::MatConst
          ? b->make(opcode<OpCode::MatRow>, Orientation::plain, Router::id(m))
          : Router::id(m);
  return {*b, b->make(opcode<OpCode::MatVec>, orientation,
                      {.matrix = rows, .vec = Router::id(v)})};
}

// A transposed matrix usable only on the left of %, where it builds
// gemv(M, v, true) rather than a node of its own.
template <impl::Numeric T> struct Transposed {
  Expression<T> m;

  [[nodiscard]] friend constexpr Expression<T>
  operator%(const Transposed &t, const Expression<T> &v) {
    return gemv(t.m, v, Orientation::transposed);
  }
};

// v + s e_i, which adds a scalar to element i; this is the adjoint of v[i].
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T>
add_at(const Expression<T> &v, const Expression<T> &s, std::size_t i) {
  if (const auto why = not_a_vector(v)) {
    return refused<T>(*why);
  }
  if (const auto why = Router::why(s)) {
    return refused<T>(*why);
  }
  if (i >= v.size() || !Router::scalar(s)) {
    return refused<T>(errc::tensor_shape);
  }
  Arena<T> &b = *Router::arena(v);
  return {b,
          b.make(opcode<OpCode::AddAt>, Element{static_cast<std::uint32_t>(i)},
                 {.vec = Router::id(v), .scalar = s.id(b)})};
}
} // namespace detail

template <impl::Numeric T = double>
[[nodiscard]] Expression<T> pack(std::span<const Expression<T>> scalars) {
  detail::Arena<T> *const b = detail::arena_of(scalars);
  return b == nullptr ? detail::refused<T>(errc::outside_equation)
                      : detail::pack(*b, scalars);
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> pack(std::initializer_list<Expression<T>> scalars) {
  return pack(std::span<const Expression<T>>{scalars});
}
template <impl::Numeric T>
[[nodiscard]] Expression<T> pack(const std::vector<Expression<T>> &scalars) {
  return pack(std::span<const Expression<T>>{scalars});
}

// A vector of n symbols named name[0] through name[n-1]; vec_param makes
// parameters, which are never differentiated.
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> vec(std::string_view name, std::size_t n) {
  return detail::in_arena<T>(
      [&](detail::Arena<T> &b) { return detail::vec(b, name, n); });
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> vec_param(std::string_view name, std::size_t n) {
  return detail::in_arena<T>(
      [&](detail::Arena<T> &b) { return detail::vec_param(b, name, n); });
}

// A constant vector or a column-major matrix.
template <impl::Numeric T = double>
[[nodiscard]] Expression<T>
vec_constant(std::span<const std::type_identity_t<T>> data) {
  return detail::in_arena<T>(
      [&](detail::Arena<T> &b) { return detail::vec_constant(b, data); });
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T> vec_constant(std::initializer_list<T> data) {
  return vec_constant(std::span<const T>{data});
}
template <impl::Numeric T = double>
[[nodiscard]] Expression<T>
mat_constant(std::size_t rows, std::size_t cols,
             std::span<const std::type_identity_t<T>> data) {
  return detail::in_arena<T>([&](detail::Arena<T> &b) {
    return detail::mat_constant(b, rows, cols, data);
  });
}

// Reductions to a scalar.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> sum(const Expression<T> &v) {
  if (const auto why = detail::not_a_vector(v)) {
    return detail::refused<T>(*why);
  }
  detail::Arena<T> &b = *detail::Router::arena(v);
  return {b, b.make(OpCode::Sum, detail::Router::id(v))};
}
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T> dot(const Expression<T> &u,
                                          const Expression<T> &v) {
  using detail::Router;
  if (const auto why = detail::poisoned(u, v)) {
    return detail::refused<T>(*why);
  }
  // Either side may be a literal, which takes the other side's width.
  detail::Arena<T> *const b =
      Router::arena(u) ? Router::arena(u) : Router::arena(v);
  const detail::Shape shape = u.size() ? Router::shape(u) : Router::shape(v);
  if (b == nullptr || shape.size() == 0 ||
      (!Router::pending(u) && !Router::pending(v) &&
       Router::arena(u) != Router::arena(v)) ||
      (u.size() != 0 && v.size() != 0 &&
       Router::shape(u) != Router::shape(v)) ||
      (!Router::pending(u) && u.size() == 0) ||
      (!Router::pending(v) && v.size() == 0)) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return {*b, b->make(detail::opcode<OpCode::Dot>,
                      {.left = u.id(*b, shape), .right = v.id(*b, shape)})};
}

// M v, or M^T v when transposed, for a constant table or a computed matrix.
template <impl::Numeric T>
[[nodiscard]] constexpr Expression<T>
gemv(const Expression<T> &m, const Expression<T> &v, bool transposed = false) {
  return detail::gemv(m, v,
                      transposed ? detail::Orientation::transposed
                                 : detail::Orientation::plain);
}

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
    detail::Arena<T> *const b = detail::Router::arena(e);
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
  using detail::Router;
  if (const auto why = detail::not_a_vector(u, w)) {
    return detail::refused<T>(*why);
  }
  detail::Arena<T> *const b = Router::arena(u);
  if (Router::arena(w) != b || Router::shape(u).cols != 1 ||
      Router::shape(w).cols != 1) {
    return detail::refused<T>(errc::tensor_shape);
  }
  return {*b, b->make(detail::opcode<OpCode::Outer>,
                      {.left = Router::id(u), .right = Router::id(w)})};
}

} // namespace ddx::rt
