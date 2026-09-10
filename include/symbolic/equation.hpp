#pragma once
#include "ops/scalar.hpp"
#include "symbolic/bound.hpp"
#include "symbolic/format.hpp"
#include "symbolic/simplify.hpp"
#include "symbolic/sweep.hpp"
#include "symbolic/symbol.hpp"
#include "util/config.hpp" // DDX_KEYED_ACCESSORS
#include "util/scope_guard.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <string_view>
#include <utility>

namespace ddx::impl {

namespace detail {
// Evaluate a tuple of expressions at one point, in canonical symbol order.
template <CSymbolList Syms>
constexpr auto eval_all(const CNumericBuffer auto &vals,
                        const CExpression auto &...es) noexcept {
  return std::array{es.template eval_seeded<Syms>(vals)...};
}

// The expressions are final by now, so this is where commutative operands get
// ordered; folding already happened as they were built.
template <CExpression E>
using canonical_t = decltype(canonicalise(std::declval<const E &>()));

template <CSymbol... Syms>
constexpr auto make_derivatives(mp::mp_list<Syms...>,
                                const CExpression auto &expr) noexcept {
  // thaw_partials: the hold that made the column is lifted again, or the row
  // it returns would differentiate to zero -- see Freeze in expressions.hpp.
  return std::tuple(canonicalise(thaw_partials(
      make_all_constant_except<Syms::value>(expr).derivative()))...);
}

template <CSymbol... Syms, CExpression... Exprs>
constexpr auto make_jac_rows(const std::tuple<Exprs...> &es,
                             mp::mp_list<Syms...> = {}) noexcept {
  return std::apply(
      [](const auto &...exprs) noexcept {
        return std::make_tuple(
            make_derivatives(mp::mp_list<Syms...>{}, exprs)...);
      },
      es);
}

} // namespace detail

template <CExpression TFirst, CExpression... TRest>
  requires((CSameValueType<TFirst, TRest> && ...))
class Equation<TFirst, TRest...> {
public:
  using value_type = typename TFirst::value_type;
  using symbols = tuple_union_t<extract_symbols_from_expr_t<TFirst>,
                                extract_symbols_from_expr_t<TRest>...>;

  static constexpr std::size_t output_dim = 1 + sizeof...(TRest);
  static constexpr std::size_t input_dim = mp::mp_size<symbols>::value;
  static constexpr std::size_t number_of_derivatives = input_dim;

private:
  // Derived from what the user wrote, so a partial that folds away a symbol
  // cannot shrink the point.
  using Exprs =
      std::tuple<detail::canonical_t<TFirst>, detail::canonical_t<TRest>...>;
  Exprs expressions;

  using point_t = std::array<value_type, input_dim>;

  // strip_seed, so a dual-valued expression answers with the derivative proper
  // rather than one still carrying its seed.  dual_scalar_t is the identity on
  // a plain scalar.
  using jacobian_tensor_t =
      md_tensor<dual_scalar_t<value_type>,
                md::extents<std::size_t, output_dim, input_dim>>;

  // One derivative row at a point, the seed stripped.
  [[nodiscard]] static constexpr auto eval_row(const auto &row,
                                               const point_t &vals) noexcept {
    return detail::strip_seed(std::apply(
        [&](const auto &...ds) {
          return detail::eval_all<symbols>(vals, ds...);
        },
        row));
  }

  [[nodiscard]] constexpr auto
  jacobian_symbolic(const point_t &vals) const noexcept
    requires(output_dim > 1 && input_dim > 0)
  {
    jacobian_tensor_t J{};
    const auto rows = jacobian_rows();
    static_for<output_dim>([&]<std::size_t I>() {
      assign_row(J, I, eval_row(std::get<I>(rows), vals));
    });
    return J;
  }

  [[nodiscard]] constexpr auto
  jacobian_reverse_mode(const point_t &vals) const noexcept
    requires(output_dim > 1 && input_dim > 0)
  {
    jacobian_tensor_t J{};
    static_for<output_dim>([&]<std::size_t I>() {
      // reverse_sweep fills a plain array; the row lands in the tensor after.
      std::array<value_type, input_dim> row{};
      reverse_sweep<symbols>(std::get<I>(expressions), vals, row);
      assign_row(J, I, detail::strip_seed(row));
    });
    return J;
  }

  // Row 0, without the leading axis.
  [[nodiscard]] constexpr auto symbolic_row(const point_t &vals) const noexcept
    requires(output_dim == 1 && input_dim > 0)
  {
    return eval_row(std::get<0>(jacobian_rows()), vals);
  }

  // The dual level carries the tangent seed while the sweep carries the
  // adjoints, so one sweep is an exact Hessian-vector product.
  // Ref: Pearlmutter, Neural Computation 6(1) (1994) 147.
  [[nodiscard]] constexpr auto hessian_forward_over_reverse(
      const std::array<dual_scalar_t<value_type>, input_dim> &values)
      const noexcept
    requires(DualLike<value_type> && input_dim > 0)
  {
    using S = dual_scalar_t<value_type>;
    nd_stack_t<S, output_dim, input_dim, 2> H{};

    // Only the seeded tangent moves.
    point_t seeds{};
    std::ranges::transform(values, seeds.begin(),
                           [](const S &v) { return value_type{v, S{}}; });

    for (auto &&[index, seeded] : std::views::enumerate(seeds)) {
      const auto j = static_cast<std::size_t>(index);
      const auto seed = scoped_seed<1>(seeded.deriv());
      static_for<output_dim>([&]<std::size_t K>() {
        point_t grads{};
        reverse_sweep<symbols>(std::get<K>(expressions), seeds, grads);

        const auto column =
            grads | std::views::transform([](const value_type &g) {
              return g.template get<1>();
            });
        for (auto &&[i, entry] :
             std::views::zip(std::views::iota(0uz, input_dim), column)) {
          H[K, i, j] = entry;
        }
      });
    }
    return H;
  }

  template <std::size_t Order>
  [[nodiscard]] constexpr auto equation_derivative_tensor_impl(
      const std::array<scalar_base_t<value_type>, input_dim> &values)
      const noexcept
    requires(input_dim > 0 && Order > 0)
  {
    using S = scalar_base_t<value_type>;
    using U = nth_dual_t<S, Order>;
    nd_stack_t<S, output_dim, input_dim, Order> result{};

    for (const auto &idx : detail::simplex_index_table_v<input_dim, Order>) {

      const auto seeds = detail::mixed_seeds<S, Order>(values, idx);

      static_for<output_dim>([&]<std::size_t OUT>() {
        const U val =
            std::get<OUT>(expressions).template eval_seeded<symbols>(seeds);
        const auto stacked = [&]<std::size_t... K>(std::index_sequence<K...>) {
          return std::array<std::size_t, Order + 1>{OUT, idx[K]...};
        }(std::make_index_sequence<Order>{});
        result.at_index(stacked) = detail::extract_nth<Order>(val);
      });
    }
    return result;
  }

  // The symbolic trees and their partials.  Every other member reduces them to
  // numbers at a point; the formatter is what wants the trees themselves.
  friend struct ::std::formatter<Equation<TFirst, TRest...>, char>;

  [[nodiscard]] constexpr const Exprs &functions() const noexcept {
    return expressions;
  }
  // On demand: storing it would instantiate the whole symbolic Jacobian every
  // time Equation<E> is named.
  [[nodiscard]] constexpr auto jacobian_rows() const noexcept {
    return detail::make_jac_rows(expressions, symbols{});
  }

  template <std::size_t N>
  [[nodiscard]] static constexpr decltype(auto) slot(auto &&self) noexcept {
    if constexpr (N == 0) {
      return std::get<0>(DDX_FWD(self).expressions);
    } else {
      // Built on demand, so this returns a value, not a dangling reference.
      auto row = std::get<0>(self.jacobian_rows());
      std::tuple_element_t<N - 1, decltype(row)> d = std::get<N - 1>(row);
      return d;
    }
  }

public:
  // explicit, or `Equation<E> eq = expr;` is ambiguous with
  // EquationConvertible's conversion operator.
  constexpr explicit Equation(TFirst first, TRest... rest) noexcept
      : expressions{detail::canonicalise(first),
                    detail::canonicalise(rest)...} {}

  // A point spelled as a range answers with result<T>, its length being unknown
  // until it arrives; every other spelling is counted by a static_assert.
  // detail::with_point is the one place that fork lives.
  template <Numeric U = value_type>
  [[nodiscard]] static constexpr auto
  point(const CEvalArg auto &...args) noexcept {
    return detail::make_point<symbols, U, input_dim>(args...);
  }

  [[nodiscard]] constexpr auto
  evaluate(const CEvalArg auto &...args) const noexcept {
    return detail::with_point<symbols, value_type, input_dim>(
        [&](const auto &vals) {
          if constexpr (output_dim == 1) {
            return std::get<0>(expressions).template eval_seeded<symbols>(vals);
          } else {
            return std::apply(
                [&](const auto &...es) {
                  return detail::eval_all<symbols>(vals, es...);
                },
                expressions);
          }
        },
        args...);
  }

  // Slot 0 is the expression itself; slot k>0 is d/d(k-1 th symbol), in
  // canonical symbol order.  Both spellings; see DDX_KEYED_ACCESSORS.
  DDX_KEYED_ACCESSORS(std::size_t N, std::size_t N, N, idx_t<N>,
                      requires(output_dim == 1 && N <= input_dim))

  // Symbolic evaluates the stored partial trees; Reverse never builds them.
  // The leading output axis appears only with more than one output, as in
  // hessian() and derivative_tensor().  Every branch sweeps `symbols`, the
  // list the point is laid out by: the canonicalised tree's own list can be
  // shorter -- (y*x)/(x*y) folds to 1 and loses both -- and a sweep indexed by
  // it would put z's partial in x's slot.
  template <DiffMode Mode = DiffMode::Reverse>
  [[nodiscard]] constexpr auto
  jacobian(const CEvalArg auto &...args) const noexcept
    requires(input_dim > 0)
  {
    return detail::with_point<symbols, value_type, input_dim>(
        [&](const auto &vals) {
          if constexpr (output_dim == 1) {
            if constexpr (Mode == DiffMode::Symbolic) {
              return symbolic_row(vals);
            } else {
              std::array<value_type, input_dim> row{};
              reverse_sweep<symbols>(std::get<0>(expressions), vals, row);
              return detail::strip_seed(row);
            }
          } else if constexpr (Mode == DiffMode::Symbolic) {
            return jacobian_symbolic(vals);
          } else {
            return jacobian_reverse_mode(vals);
          }
        },
        args...);
  }

  // No DiffMode: only the reverse path.  A symbolic second derivative would be
  // derivative_tensor<2> with extra steps, and the colouring is what makes this
  // worth having over that.
  [[nodiscard]] constexpr auto
  hessian(const CEvalArg auto &...args) const noexcept
    requires(DualLike<value_type> && input_dim > 0)
  {
    return detail::with_point<symbols, dual_scalar_t<value_type>, input_dim>(
        [&](const auto &vals) {
          if constexpr (output_dim == 1) {
            return detail::reverse_mode_hessian<symbols>(
                std::get<0>(expressions), vals);
          } else {
            return hessian_forward_over_reverse(vals);
          }
        },
        args...);
  }

  template <std::size_t Order>
  [[nodiscard]] constexpr auto
  derivative_tensor(const CEvalArg auto &...args) const noexcept
    requires(input_dim > 0 && Order > 0)
  {
    return detail::with_point<symbols, scalar_base_t<value_type>, input_dim>(
        [&](const auto &vals) {
          if constexpr (output_dim == 1) {
            return detail::derivative_tensor_impl<Order>(
                std::get<0>(expressions), vals);
          } else {
            return equation_derivative_tensor_impl<Order>(vals);
          }
        },
        args...);
  }

  // A plain number, not a one-entry tensor.  Spelled over a plain scalar, so it
  // has to be guarded rather than disappearing with forward mode.
  template <std::size_t Order>
  [[nodiscard]] DDX_ALWAYS_INLINE constexpr auto
  univariate_derivative(scalar_base_t<value_type> x0) const noexcept
    requires(input_dim == 1 && output_dim == 1 && Order > 0)
  {
    return detail::univariate_derivative_impl<Order>(std::get<0>(expressions),
                                                     x0);
  }
};

template <CExpression T, CExpression... Ts>
Equation(T, Ts...) -> Equation<T, Ts...>;

// Declared back in expr/expressions.hpp, where Equation was still incomplete.
template <typename Derived>
template <typename Eq>
  requires std::same_as<Eq, Equation<Derived>>
constexpr EquationConvertible<Derived>::operator Eq() const noexcept {
  return Equation<Derived>{static_cast<const Derived &>(*this)};
}

template <CExpression... Ts>
std::ostream &operator<<(std::ostream &out, const Equation<Ts...> &eq) {
  return out << std::format("{}", eq);
}

} // namespace ddx::impl

// One block per output: the function, then its Jacobian row in canonical symbol
// order.  The spec is forwarded to every expression printed.
template <ddx::impl::CExpression... Ts>
struct std::formatter<ddx::impl::Equation<Ts...>, char> {
  constexpr auto parse(std::format_parse_context &ctx) {
    // (iterator, sentinel): the iterator is const char* on libstdc++ but a
    // class type on MSVC.
    spec_ = std::string_view(ctx.begin(), ctx.end());
    if (const auto close = spec_.find('}'); close != std::string_view::npos) {
      spec_ = spec_.substr(0, close);
    }
    return ctx.begin() + static_cast<std::ptrdiff_t>(spec_.size());
  }

  auto format(const ddx::impl::Equation<Ts...> &eq,
              std::format_context &ctx) const {
    using Eq = ddx::impl::Equation<Ts...>;
    const std::string one = std::format("{{:{}}}", spec_);
    auto out = ctx.out();

    const auto rows = eq.jacobian_rows();
    ddx::impl::static_for<Eq::output_dim>([&]<std::size_t I>() {
      out = std::format_to(out, "f{}: ", I);
      out = std::vformat_to(out, one,
                            std::make_format_args(std::get<I>(eq.functions())));
      out = std::format_to(out, "\n  jac: ");
      const auto &row = std::get<I>(rows);
      ddx::impl::static_for<Eq::input_dim>([&]<std::size_t J>() {
        if constexpr (J > 0) {
          out = std::format_to(out, ", ");
        }
        out =
            std::vformat_to(out, one, std::make_format_args(std::get<J>(row)));
      });
      out = std::format_to(out, "\n");
    });
    return out;
  }

private:
  std::string_view spec_{};
};
