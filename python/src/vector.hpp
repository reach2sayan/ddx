#pragma once

#include "error.hpp"
#include "expression.hpp"
#include "rt/tensor.hpp"

#include "util/ranges.hpp"

#include <cstddef>
#include <format>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

// The vector and matrix handles Python holds: an RTVector or RTMatrix that owns
// its arena, as PyExpression does, and for the same reason.  A poisoned vector
// raises here, at the call that poisoned it, rather than reaching equation().
namespace ddx::py {

class PyMatrix;

class PyVector {
public:
  using Base = rt::RTVector<double>;
  using Arena = PyExpression::Arena;

  PyVector() = default;
  PyVector(Base v, Arena a) : vector_(v), arena_(std::move(a)) {
    if (const auto why = rt::detail::Tensoring::why(vector_)) {
      fail_with(*why);
    }
  }

  [[nodiscard]] std::size_t size() const noexcept { return vector_.size(); }
  [[nodiscard]] PyExpression at(std::size_t i) const {
    if (i >= size()) {
      fail_with(errc::tensor_shape);
    }
    return {vector_[i], arena_};
  }
  [[nodiscard]] std::string repr() const {
    return std::format("Vector({} #{})", size(), vector_.id());
  }
  [[nodiscard]] const Base &base() const noexcept { return vector_; }
  [[nodiscard]] const Arena &arena() const noexcept { return arena_; }

  // The vector of `scalars`, in the arena they name, else `fallback` -- the
  // model running now.  Here rather than in the module: a scalar's arena is
  // this class's to read.
  [[nodiscard]] static PyVector pack(std::span<const PyExpression> scalars,
                                     const Arena &fallback) {
    Arena arena = fallback;
    for (const PyExpression &e : scalars) {
      if (e.arena()) {
        arena = e.arena();
        break;
      }
    }
    if (!arena) {
      fail_with(errc::no_arena);
    }
    const auto bases = scalars | std::views::transform(&PyExpression::base) |
                       impl::to<std::vector<rt::RTExpression<double>>>();
    return {rt::pack(*arena, std::span<const rt::RTExpression<double>>{bases}),
            arena};
  }
  [[nodiscard]] static PyVector add_at(const PyVector &v,
                                       const PyExpression &s, std::size_t i) {
    return {rt::add_at(v.vector_, s.base(), i), v.arena_};
  }
  [[nodiscard]] static PyVector gemv(const PyMatrix &m, const PyVector &v,
                                     bool transposed);

  // The base's own composition, with the arena put back; mixed operands take
  // the vector's arena, since a scalar may be a pending literal.
  friend PyVector vcombine(const PyVector &l, const PyVector &r, auto &&fn) {
    return {fn(l.vector_, r.vector_), l.arena_ ? l.arena_ : r.arena_};
  }
  friend PyVector vmap(rt::OpCode op, const PyVector &u) {
    return {Base::map(op, u.vector_), u.arena_};
  }
  friend PyVector vzip(rt::OpCode op, const PyVector &l, const PyVector &r) {
    return {Base::zip(op, l.vector_, r.vector_),
            l.arena_ ? l.arena_ : r.arena_};
  }
  friend PyVector vscale(const PyExpression &s, const PyVector &v) {
    return {Base::scale(s.base(), v.vector_), v.arena_};
  }
  // A reduction to a scalar raises where it is formed, as a vector does.
  friend PyExpression reduce(const PyVector &v, auto &&fn) {
    return checked(fn(v.vector_), v.arena_);
  }
  friend PyExpression reduce(const PyVector &u, const PyVector &v, auto &&fn) {
    return checked(fn(u.vector_, v.vector_), u.arena_ ? u.arena_ : v.arena_);
  }

private:
  [[nodiscard]] static PyExpression checked(const rt::RTExpression<double> &e,
                                            Arena arena) {
    if (const auto why = rt::detail::Rooting::why(e)) {
      fail_with(*why);
    }
    return {e, std::move(arena)};
  }

  Base vector_;
  Arena arena_;
};

class PyMatrix {
public:
  using Base = rt::RTMatrix<double>;
  using Arena = PyExpression::Arena;

  PyMatrix(Base m, Arena a) : matrix_(m), arena_(std::move(a)) {
    if (matrix_.poisoned()) {
      fail_with(errc::tensor_shape);
    }
  }
  [[nodiscard]] std::size_t rows() const noexcept { return matrix_.rows(); }
  [[nodiscard]] std::size_t cols() const noexcept { return matrix_.cols(); }
  [[nodiscard]] std::string repr() const {
    return std::format("Matrix({}x{} #{})", rows(), cols(), matrix_.id());
  }
  [[nodiscard]] const Base &base() const noexcept { return matrix_; }
  [[nodiscard]] const Arena &arena() const noexcept { return arena_; }

private:
  Base matrix_;
  Arena arena_;
};

inline PyVector PyVector::gemv(const PyMatrix &m, const PyVector &v,
                               bool transposed) {
  return {transposed ? rt::gemv_t(m.base(), v.vector_)
                     : rt::gemv(m.base(), v.vector_),
          v.arena_};
}

} // namespace ddx::py
