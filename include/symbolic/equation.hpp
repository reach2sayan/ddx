#pragma once
#include "dual/taylor_dual.hpp" // the univariate sweep's scalar
#include "md/tensor.hpp"
#include "ops/mode.hpp" // DiffMode
#include "ops/scalar.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/interpret.hpp"
#include "symbolic/bound.hpp"
#include "symbolic/format.hpp"
#include "symbolic/simplify.hpp"
#include "symbolic/symbol.hpp"
#include "util/config.hpp" // DDX_KEYED_ACCESSORS
#include "util/scope_guard.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

// A compile-time expression and its numbers.  The tree is the type: what is
// kept here is the canonicalised expressions and, on demand, their symbolic
// partial trees.  Every number comes from the runtime graph the expressions
// lower to -- one arena per call, swept at whatever scalar the point is
// spelled in: a dual for a derivative tensor, a truncated polynomial for the
// univariate derivative.  A model swept in a loop goes through
// rt::equation(eq), which lowers once and keeps lanes, kernels and the
// compiled code.
namespace ddx::impl {

namespace detail {

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

// A dual-valued row carries the seed in its dual part; the derivative proper
// is the real part.
template <Numeric T, std::size_t N>
[[nodiscard]] constexpr auto
strip_seed(const std::array<T, N> &grads) noexcept {
  if constexpr (DualLike<T>) {
    std::array<dual_scalar_t<T>, N> result{};
    std::ranges::transform(grads, result.begin(),
                           [](const T &g) { return g.template get<0>(); });
    return result;
  } else {
    return grads;
  }
}
template <Numeric T> [[nodiscard]] constexpr auto strip_seed(const T &v) {
  if constexpr (DualLike<T>) {
    return v.template get<0>();
  } else {
    return v;
  }
}

template <Numeric S, std::size_t Depth>
constexpr nth_dual_t<S, Depth> make_mixed_seed(S value,
                                               std::span<const std::size_t> idx,
                                               std::size_t k) noexcept {
  if constexpr (Depth == 0) {
    return value;
  } else if constexpr (Depth == 1) {
    return nth_dual_t<S, 1>{value, k == idx[0] ? S{1} : S{}};
  } else {
    auto inner =
        make_mixed_seed<S, Depth - 1>(std::move(value), idx.subspan(1), k);
    auto outer_tangent = embed_constant<S, Depth - 1>(k == idx[0] ? S{1} : S{});
    return nth_dual_t<S, Depth>{std::move(inner), std::move(outer_tangent)};
  }
}

// The cartesian product of Order copies of [0, N), in layout_right order.
template <std::size_t N, std::size_t Order>
[[nodiscard]] constexpr auto index_grid() noexcept {
  return index_apply<Order>([]<std::size_t... D>() {
    return std::views::cartesian_product(
        ((void)D, std::views::iota(std::size_t{0}, N))...);
  });
}

// One representative per permutation class: by Schwarz only C(N+Order-1, Order)
// of the N^Order entries are distinct.  A consteval table, not a filtered view
// or lazy iterator: both measured 4x slower.
// Ref: Neidinger, ACM TOMS 18(2) (1992) 159.
template <std::size_t N, std::size_t Order>
  requires(N > 0 && Order > 0)
consteval auto simplex_index_table() noexcept {
  std::array<std::array<std::size_t, Order>, binomial(N + Order - 1, Order)>
      out{};
  std::array<std::size_t, Order> idx{};
  for (auto &slot : out) {
    slot = idx;
    // Raise the rightmost position that can be, and collapse the rest onto it.
    for (std::size_t p = Order; p-- > 0;) {
      if (idx[p] + 1 < N) {
        const std::size_t v = ++idx[p];
        std::ranges::fill(idx | std::views::drop(p + 1), v);
        break;
      }
    }
  }
  return out;
}

template <std::size_t N, std::size_t Order>
inline constexpr auto simplex_index_table_v = simplex_index_table<N, Order>();

template <std::size_t N, Numeric T>
constexpr auto extract_nth(const T &x) noexcept {
  return component<N, 1>(x);
}

// Every variable lifted to nth_dual_t, with the tangents `idx` names on.
template <Numeric S, std::size_t Order, std::size_t N>
[[nodiscard]] constexpr std::array<nth_dual_t<S, Order>, N>
mixed_seeds(const std::array<S, N> &values,
            std::span<const std::size_t> idx) noexcept {
  std::array<nth_dual_t<S, Order>, N> seeds{};
  std::ranges::transform(values, std::views::iota(0uz, N), seeds.begin(),
                         [&idx](const S &v, std::size_t k) {
                           return make_mixed_seed<S, Order>(v, idx, k);
                         });
  return seeds;
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
  // The scalar the graph is built over; a dual-valued expression lowers to a
  // graph over its base and is swept at the dual.
  using scalar = scalar_base_t<value_type>;
  using S = dual_scalar_t<value_type>;

  // Derived from what the user wrote, so a partial that folds away a symbol
  // cannot shrink the point.
  using Exprs =
      std::tuple<detail::canonical_t<TFirst>, detail::canonical_t<TRest>...>;
  Exprs expressions;

  using point_t = std::array<value_type, input_dim>;
  using jacobian_tensor_t =
      md_tensor<S, md::extents<std::size_t, output_dim, input_dim>>;

  // The runtime graph of the expressions: the symbols declared first, in the
  // Equation's order, so a column is a symbol's slot in the point -- the
  // canonicalised tree's own list can be shorter, (y*x)/(x*y) folds to 1 and
  // loses both.
  struct Lowered {
    rt::Builder<scalar> arena;
    std::array<rt::NodeId, output_dim> roots;
  };
  [[nodiscard]] constexpr Lowered lower() const {
    Lowered l{};
    mp::mp_for_each<symbols>(
        [&l]<typename Sym>(Sym) { (void)rt::var(l.arena, Sym::name); });
    static_for<output_dim>([&]<std::size_t I>() {
      l.roots[I] = rt::to_graph(l.arena, std::get<I>(expressions)).id(l.arena);
    });
    return l;
  }
  // The nodes `wanted` reach, at the point's scalar.
  template <Numeric U>
  [[nodiscard]] static constexpr std::vector<U>
  sweep(const Lowered &l, std::span<const rt::NodeId> wanted,
        const std::array<U, input_dim> &at) {
    return rt::evaluate_reachable(l.arena, wanted, at);
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
        [&](const point_t &vals) {
          const Lowered l = lower();
          const auto v = sweep(l, l.roots, vals);
          if constexpr (output_dim == 1) {
            return v[l.roots[0]];
          } else {
            return index_apply<output_dim>([&]<std::size_t... I>() {
              return std::array<value_type, output_dim>{v[l.roots[I]]...};
            });
          }
        },
        args...);
  }

  // Slot 0 is the expression itself; slot k>0 is d/d(k-1 th symbol), in
  // canonical symbol order.  Both spellings; see DDX_KEYED_ACCESSORS.
  DDX_KEYED_ACCESSORS(std::size_t N, std::size_t N, N, idx_t<N>,
                      requires(output_dim == 1 && N <= input_dim))

  // Symbolic lowers the stored partial trees beside the function and sweeps
  // them; Reverse builds the adjoint sweep on the graph and never forms a
  // tree.  The leading output axis appears only with more than one output, as
  // in hessian() and derivative_tensor().
  template <DiffMode Mode = DiffMode::Reverse>
  [[nodiscard]] constexpr auto
  jacobian(const CEvalArg auto &...args) const noexcept
    requires(input_dim > 0)
  {
    return detail::with_point<symbols, value_type, input_dim>(
        [&](const point_t &vals) {
          Lowered l = lower();
          std::array<rt::NodeId, output_dim * input_dim> cells{};
          if constexpr (Mode == DiffMode::Symbolic) {
            const auto rows = jacobian_rows();
            static_for<output_dim>([&]<std::size_t I>() {
              static_for<input_dim>([&]<std::size_t J>() {
                cells[I * input_dim + J] =
                    rt::to_graph(l.arena, std::get<J>(std::get<I>(rows)))
                        .id(l.arena);
              });
            });
          } else {
            const rt::Jacobian j = rt::build_jacobian_impl(l.arena, l.roots);
            for (const std::size_t i : std::views::iota(0uz, output_dim)) {
              for (const std::size_t k : std::views::iota(0uz, input_dim)) {
                cells[i * input_dim + k] = j.at(i, k);
              }
            }
          }
          const auto v = sweep(l, cells, vals);
          if constexpr (output_dim == 1) {
            std::array<value_type, input_dim> row{};
            std::ranges::transform(cells, row.begin(),
                                   [&v](rt::NodeId c) { return v[c]; });
            return detail::strip_seed(row);
          } else {
            jacobian_tensor_t J{};
            for (const std::size_t i : std::views::iota(0uz, output_dim)) {
              for (const std::size_t k : std::views::iota(0uz, input_dim)) {
                J[i, k] = detail::strip_seed(v[cells[i * input_dim + k]]);
              }
            }
            return J;
          }
        },
        args...);
  }

  // The reverse-mode Hessian, compressed by the graph's colouring and read
  // back dense.  The colouring is libddx's, so this is not constant-evaluated;
  // derivative_tensor<2> is.
  [[nodiscard]] auto hessian(const CEvalArg auto &...args) const noexcept
    requires(input_dim > 0)
  {
    return detail::with_point<symbols, S, input_dim>(
        [&](const std::array<S, input_dim> &vals) {
          Lowered l = lower();
          std::array<rt::Hessian, output_dim> hs;
          std::vector<rt::NodeId> wanted;
          for (const std::size_t k : std::views::iota(0uz, output_dim)) {
            hs[k] = rt::build_hessian_impl(l.arena, l.roots[k]);
            append(wanted, hs[k].compressed);
          }
          const auto v = sweep(l, wanted, vals);
          constexpr auto dims = std::views::iota(0uz, input_dim);
          if constexpr (output_dim == 1) {
            nd_tensor_t<S, input_dim, 2> H{};
            for (const auto [i, j] : std::views::cartesian_product(dims, dims)) {
              H[i, j] = v[hs[0].at(i, j)];
            }
            return H;
          } else {
            nd_stack_t<S, output_dim, input_dim, 2> H{};
            for (const std::size_t k : std::views::iota(0uz, output_dim)) {
              for (const auto [i, j] :
                   std::views::cartesian_product(dims, dims)) {
                H[k, i, j] = v[hs[k].at(i, j)];
              }
            }
            return H;
          }
        },
        args...);
  }

  // Forward mode at a dual nested Order deep: one sweep per distinct
  // multi-index serves every output.
  template <std::size_t Order>
  [[nodiscard]] constexpr auto
  derivative_tensor(const CEvalArg auto &...args) const noexcept
    requires(input_dim > 0 && Order > 0)
  {
    using U = nth_dual_t<scalar, Order>;
    return detail::with_point<symbols, scalar, input_dim>(
        [&](const std::array<scalar, input_dim> &vals) {
          const Lowered l = lower();
          auto result = [] {
            if constexpr (output_dim == 1) {
              return nd_tensor_t<scalar, input_dim, Order>{};
            } else {
              return nd_stack_t<scalar, output_dim, input_dim, Order>{};
            }
          }();
          for (const auto &idx :
               detail::simplex_index_table_v<input_dim, Order>) {
            const auto seeds = detail::mixed_seeds<scalar, Order>(vals, idx);
            const auto v = sweep<U>(l, l.roots, seeds);
            if constexpr (output_dim == 1) {
              result.at_index(idx) = detail::extract_nth<Order>(v[l.roots[0]]);
            } else {
              for (const std::size_t k : std::views::iota(0uz, output_dim)) {
                const auto stacked =
                    [&]<std::size_t... K>(std::index_sequence<K...>) {
                      return std::array<std::size_t, Order + 1>{k, idx[K]...};
                    }(std::make_index_sequence<Order>{});
                result.at_index(stacked) =
                    detail::extract_nth<Order>(v[l.roots[k]]);
              }
            }
          }
          return result;
        },
        args...);
  }

  // One Taylor sweep, then undo the 1/Order! normalisation.  A plain number,
  // not a one-entry tensor.  Ref: Griewank, Utke & Walther, Math. Comp.
  // 69(231) (2000) 1117.
  template <std::size_t Order>
  [[nodiscard]] DDX_ALWAYS_INLINE constexpr auto
  univariate_derivative(scalar x0) const noexcept
    requires(input_dim == 1 && output_dim == 1 && Order > 0)
  {
    using Taylor = TaylorDual<scalar, Order>;
    static_assert(rt::detail::computes_at_tree<
                      Taylor, std::tuple_element_t<0, Exprs>>(),
                  "an operation the Taylor arithmetic does not define");
    const Lowered l = lower();
    const auto v = sweep(l, l.roots, std::array<Taylor, 1>{Taylor::variable(x0)});
    return v[l.roots[0]].c[Order] *
           static_cast<scalar>(detail::compile_time_factorial(Order));
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
