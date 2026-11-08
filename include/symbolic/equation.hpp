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
#include "util/config.hpp" // DDX_FWD
#include "util/meta.hpp"
#include "util/scope_guard.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

// A compile-time equation. Each call lowers the expressions into a fresh rt
// arena, so a model evaluated in a loop belongs in rt::Equation instead.
namespace ddx::impl {

namespace detail {

// Commutative operands are ordered here, once the expressions are final.
template <CExpression E>
using canonical_t = decltype(canonicalise(std::declval<const E &>()));

// thaw_partials lifts the hold that made each column, or the returned row
// would differentiate to zero. See Freeze in expressions.hpp.
constexpr auto make_derivatives(CSymbolList auto syms,
                                const CExpression auto &expr) noexcept {
  return meta::unpack(syms, [&](auto... s) noexcept {
    return std::tuple(canonicalise(
        thaw_partials(make_all_constant_except<s.value>(expr).derivative()))...);
  });
}

template <CExpression... Exprs>
constexpr auto make_jac_rows(const std::tuple<Exprs...> &es,
                             CSymbolList auto syms) noexcept {
  return meta::unpack(es, [&](const auto &...exprs) noexcept {
    return std::make_tuple(make_derivatives(syms, exprs)...);
  });
}

// A dual-valued row carries the seed in its dual part, so the derivative is
// the real part.
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

// One sorted multi-index per symmetry class, since by Schwarz only
// C(N+Order-1, Order) entries differ. A lazy view measured 4x slower.
// Ref: Neidinger, ACM TOMS 18(2) (1992) 159.
template <std::size_t N, std::size_t Order>
  requires(N > 0 && Order > 0)
consteval auto simplex_index_table() noexcept {
  std::array<std::array<std::size_t, Order>, binomial(N + Order - 1, Order)>
      out{};
  std::array<std::size_t, Order> idx{};
  for (auto &slot : out) {
    slot = idx;
    // Raise the rightmost index that can grow, and set those after it equal.
    for (const std::size_t p : std::views::iota(0uz, Order) | std::views::reverse) {
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

// Lifts every variable to nth_dual_t, seeding the tangents that idx names.
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
  using symbols = meta::list_t<symbol_set(expr_symbols_t<TFirst>{},
                                           expr_symbols_t<TRest>{}...)>;

  static constexpr std::size_t output_dim = 1 + sizeof...(TRest);
  static constexpr std::size_t input_dim = meta::size(symbols{});
  static constexpr std::size_t number_of_derivatives = input_dim;

private:
  // The scalar the graph is built over. A dual-valued expression lowers to a
  // graph over its base scalar and is swept at the dual.
  using scalar = scalar_base_t<value_type>;
  using S = dual_scalar_t<value_type>;

  using Exprs =
      std::tuple<detail::canonical_t<TFirst>, detail::canonical_t<TRest>...>;
  Exprs expressions;

  using point_t = std::array<value_type, input_dim>;
  using jacobian_tensor_t =
      md_tensor<S, md::extents<std::size_t, output_dim, input_dim>>;

  // Symbols are declared first, in Equation order, so a column is the symbol's
  // slot in the point even when a folded tree such as (y*x)/(x*y) loses it.
  struct Lowered {
    rt::Arena<scalar> arena;
    std::array<rt::NodeId, output_dim> roots;
  };
  [[nodiscard]] constexpr Lowered lower() const {
    Lowered l{};
    // The tree sizes bound the arena, since interning only shrinks it.
    l.arena.reserve(input_dim +
                        (node_count_v<detail::canonical_t<TFirst>> + ... +
                         node_count_v<detail::canonical_t<TRest>>),
                    input_dim);
    meta::for_each(symbols{}, [&l](auto s) { (void)rt::var(l.arena, s.name); });
    meta::for_each(meta::iota<output_dim>, [&](auto i) {
      l.roots[i] = rt::to_graph(l.arena, std::get<i>(expressions)).id(l.arena);
    });
    return l;
  }

  // The formatter is the only reader of the symbolic trees themselves.
  friend struct ::std::formatter<Equation<TFirst, TRest...>, char>;

  [[nodiscard]] constexpr const Exprs &functions() const noexcept {
    return expressions;
  }
  // Built on demand, since storing the rows would instantiate the symbolic
  // Jacobian every time Equation<E> is named.
  [[nodiscard]] constexpr auto jacobian_rows() const noexcept {
    return detail::make_jac_rows(expressions, symbols{});
  }

public:
  // get<0> is the expression itself and get<k> its partial in the (k-1)th
  // symbol, in canonical order.
  template <std::size_t N>
    requires(output_dim == 1 && N <= input_dim)
  [[nodiscard]] constexpr decltype(auto) get(this auto &&self) noexcept {
    if constexpr (N == 0) {
      return std::get<0>(DDX_FWD(self).expressions);
    } else {
      // The rows are built on demand, so the partial is returned by value.
      auto row = std::get<0>(self.jacobian_rows());
      std::tuple_element_t<N - 1, decltype(row)> d = std::get<N - 1>(row);
      return d;
    }
  }

  // Explicit, or `Equation<E> eq = expr;` is ambiguous with
  // EquationConvertible's conversion operator.
  constexpr explicit Equation(TFirst first, TRest... rest) noexcept
      : expressions{detail::canonicalise(first),
                    detail::canonicalise(rest)...} {}

  // A point given as a range returns result<T>, since its length is only known
  // at run time; every other spelling is checked by a static_assert.
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
          const auto v = rt::evaluate_reachable(l.arena, l.roots, vals);
          if constexpr (output_dim == 1) {
            return v[l.roots[0]];
          } else {
            return meta::unpack(meta::iota<output_dim>, [&](auto... i) {
              return std::array<value_type, output_dim>{v[l.roots[i]]...};
            });
          }
        },
        args...);
  }

  // Symbolic lowers the partial trees; Reverse builds the adjoint on the graph.
  // The output axis appears only with several outputs, as in hessian().
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
            meta::for_each(meta::iota<output_dim>, [&](auto i) {
              meta::for_each(meta::iota<input_dim>, [&](auto j) {
                cells[i * input_dim + j] =
                    rt::to_graph(l.arena, std::get<j>(std::get<i>(rows)))
                        .id(l.arena);
              });
            });
          } else {
            const rt::Jacobian j = rt::derive_jacobian(l.arena, l.roots);
            for (const std::size_t i : std::views::iota(0uz, output_dim)) {
              for (const std::size_t k : std::views::iota(0uz, input_dim)) {
                cells[i * input_dim + k] = j.at(i, k);
              }
            }
          }
          const auto v = rt::evaluate_reachable(l.arena, cells, vals);
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

  // The reverse-mode Hessian, compressed by colouring and read back dense. Not
  // constexpr, because the colouring lives in libddx; derivative_tensor<2> is.
  [[nodiscard]] auto hessian(const CEvalArg auto &...args) const noexcept
    requires(input_dim > 0)
  {
    return detail::with_point<symbols, S, input_dim>(
        [&](const std::array<S, input_dim> &vals) {
          Lowered l = lower();
          std::array<rt::Hessian, output_dim> hs;
          std::vector<rt::NodeId> wanted;
          for (const std::size_t k : std::views::iota(0uz, output_dim)) {
            hs[k] = rt::derive_hessian(l.arena, l.roots[k]);
            append(wanted, hs[k].compressed);
          }
          const auto v = rt::evaluate_reachable(l.arena, wanted, vals);
          constexpr auto dims = std::views::iota(0uz, input_dim);
          if constexpr (output_dim == 1) {
            nd_tensor_t<S, input_dim, 2> H{};
            for (const auto [i, j] :
                 std::views::cartesian_product(dims, dims)) {
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

  // Forward mode with duals nested Order deep: one sweep per distinct
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
          // A fresh arena holds little beyond the roots' cone, so it is walked
          // whole without a liveness pass.
          const rt::detail::ArenaWalk walk{l.arena};
          std::vector<U> v(l.arena.size()), vtape(walk.extent());
          for (const auto &idx :
               detail::simplex_index_table_v<input_dim, Order>) {
            walk(l.arena, detail::mixed_seeds<scalar, Order>(vals, idx),
                 std::span<U>{v}, std::span<U>{vtape});
            for (const auto [k, root] : l.roots | std::views::enumerate) {
              const auto at = [&] {
                if constexpr (output_dim == 1) {
                  return idx;
                } else {
                  return meta::unpack(meta::iota<Order>, [&](auto... o) {
                    return std::array{static_cast<std::size_t>(k), idx[o]...};
                  });
                }
              }();
              result.at_index(at) = detail::extract_nth<Order>(v[root]);
            }
          }
          return result;
        },
        args...);
  }

  // One Taylor sweep, then the 1/Order! normalisation undone. Returns a scalar.
  // Ref: Griewank, Utke & Walther, Math. Comp. 69(231) (2000) 1117.
  template <std::size_t Order>
  [[nodiscard]] DDX_ALWAYS_INLINE constexpr auto
  univariate_derivative(scalar x0) const noexcept
    requires(input_dim == 1 && output_dim == 1 && Order > 0)
  {
    using Taylor = TaylorDual<scalar, Order>;
    static_assert(
        rt::detail::computes_at_tree<Taylor, std::tuple_element_t<0, Exprs>>(),
        "an operation the Taylor arithmetic does not define");
    const Lowered l = lower();
    const auto v = rt::evaluate_reachable(
        l.arena, l.roots, std::array<Taylor, 1>{Taylor::variable(x0)});
    return v[l.roots[0]].c[Order] *
           static_cast<scalar>(detail::compile_time_factorial(Order));
  }
};

template <CExpression T, CExpression... Ts>
Equation(T, Ts...) -> Equation<T, Ts...>;

// Declared in expressions.hpp, where Equation was still incomplete.
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

// Prints each output and then its Jacobian row, in canonical symbol order. The
// format spec applies to every expression printed.
template <ddx::impl::CExpression... Ts>
struct std::formatter<ddx::impl::Equation<Ts...>, char> {
  constexpr auto parse(std::format_parse_context &ctx) {
    // The (iterator, sentinel) constructor, because the iterator is const char*
    // on libstdc++ but a class type on MSVC.
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
    ddx::meta::for_each(ddx::meta::iota<Eq::output_dim>, [&](auto i) {
      out = std::format_to(out, "f{}: ", i());
      out = std::vformat_to(out, one,
                            std::make_format_args(std::get<i>(eq.functions())));
      out = std::format_to(out, "\n  jac: ");
      const auto &row = std::get<i>(rows);
      ddx::meta::for_each(ddx::meta::iota<Eq::input_dim>, [&](auto j) {
        if constexpr (j > 0) {
          out = std::format_to(out, ", ");
        }
        out =
            std::vformat_to(out, one, std::make_format_args(std::get<j>(row)));
      });
      out = std::format_to(out, "\n");
    });
    return out;
  }

private:
  std::string_view spec_{};
};
