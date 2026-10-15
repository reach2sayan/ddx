#pragma once

#include "md/workspace.hpp"      // HessianStatic, symmetrize
#include "symbolic/coupling.hpp" // compile-time Hessian sparsity + colouring

// Forward mode supplies the truncated-polynomial scalar the univariate sweep
// runs on.
#include "dual/taylor_dual.hpp"
#include "md/tensor.hpp"
#include "ops/mode.hpp"
#include "ops/scalar.hpp"
#include "symbolic/entry.hpp"
#include "symbolic/expressions.hpp"
#include "symbolic/symbol.hpp"
#include "symbolic/traits.hpp"
#include "util/config.hpp"
#include "util/scope_guard.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace ddx::impl {

template <CExpression Expr>
using node_cache_t = std::array<typename std::remove_cvref_t<Expr>::value_type,
                                node_count_v<std::remove_cvref_t<Expr>>>;

// A leaf reads its slot out of `vals`, in canonical symbol order.  `Store` is
// the parent's `reads_primals`: no rule reads its own slot.
template <std::size_t Base = 0, CSymbolList Syms, bool Store = true,
          CExpression E, CNumericBuffer Vals, CNumericBuffer Cache>
constexpr auto fill_cache(const E &node, const Vals &vals,
                          Cache &cache) noexcept {
  using U = std::remove_cvref_t<E>;
  using VT = typename U::value_type;
  if constexpr (CExpressionNode<U>) {
    using Kids = typename U::children_t;
    const VT v = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return typename U::op_type::func_type{}(
          fill_cache<child_base_at<Base, Kids, I>(), Syms,
                     U::op_type::reads_primals>(std::get<I>(node.expressions()),
                                                vals, cache)...);
    }(std::make_index_sequence<std::tuple_size_v<Kids>>{});
    if constexpr (Store) {
      cache[Base] = v;
    }
    return v;
  } else { // leaf (Variable / Constant / Lit)
    const VT v = node.template eval_seeded<Syms>(vals);
    if constexpr (Store) {
      cache[Base] = v;
    }
    return v;
  }
}

// Fill the primal cache from `seeds`, then push adjoints back from a root
// adjoint of 1.  Returns the root value; `grads` receives the partials.
// Ref: Linnainmaa, BIT 16(2) (1976) 146.  docs/reverse_mode_by_example.md
// walks one of these by hand.
template <CSymbolList Syms>
DDX_ALWAYS_INLINE constexpr auto
reverse_sweep(const CExpression auto &expr, const CNumericBuffer auto &seeds,
              CNumericBuffer auto &grads) noexcept {
  using Expr = std::remove_cvref_t<decltype(expr)>;
  using T = typename Expr::value_type;
  node_cache_t<Expr> cache{};
  // Store=false: the root's value is the return value, so storing it is dead.
  const T root = fill_cache<0, Syms, false>(expr, seeds, cache);
  expr.backward(Syms{}, T{1}, grads, cache);
  return root;
}

namespace detail {

// The symbol list a sweep is indexed by: the caller's where it named one, the
// expression's own otherwise.  An Equation's list can be longer than its
// canonicalised tree's -- (y*x)/(x*y) folds to 1 and loses both -- and its
// point is laid out by the longer one.
template <typename Given, CExpression Expr>
using sweep_symbols_t =
    std::conditional_t<std::is_void_v<Given>,
                       expr_symbols_t<std::remove_cvref_t<Expr>>, Given>;

// One backward sweep per colour.  Colours are an NTTP: as an argument the loop
// bound is a runtime load, the colour loop stops unrolling, and a dense Hessian
// measured slower than a plain column sweep.
template <auto Colors, typename Given = void, CExpression Expr,
          CNumericBuffer Point, typename Harvest>
  requires std::invocable<
      Harvest &, std::size_t,
      const typename std::remove_cvref_t<Expr>::value_type &,
      std::array<typename std::remove_cvref_t<Expr>::value_type,
                 mp::mp_size<sweep_symbols_t<Given, Expr>>::value> &>
DDX_ALWAYS_INLINE constexpr void color_sweeps(const Expr &expr, const Point &x,
                                              Harvest &&harvest) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using S = dual_scalar_t<T>;
  using Syms = sweep_symbols_t<Given, Expr>;
  constexpr std::size_t N = mp::mp_size<Syms>::value;
  static_assert(Colors.count == 0 ||
                    std::ranges::max(Colors.color) + 1 == Colors.count,
                "column_coloring: count is one past the highest colour");

  // Only the seeded tangents move between colours.
  std::array<T, N> seeds{};
  std::ranges::transform(x, seeds.begin(), [](const auto &v) {
    return T{static_cast<S>(v), S{}};
  });

  // Seeds outside the colour are already zero, so only this one moves.
  static constexpr bool kIdentity =
      std::ranges::equal(Colors.color, std::views::iota(0uz, N));
  const auto toggle = [&](std::size_t c, const S &v) {
    if constexpr (kIdentity) {
      seeds[c].deriv() = v;
    } else {
      auto seed_view = std::views::zip(seeds, Colors.color) |
                       std::views::filter(
                           [c](const auto &p) { return std::get<1>(p) == c; }) |
                       std::views::keys;
      for (auto &&seed : seed_view) {
        seed.deriv() = v;
      }
    }
  };

  for (const std::size_t c : std::views::iota(0uz, Colors.count)) {
    toggle(c, S{1});
    std::array<T, N> grads{};
    const T root = reverse_sweep<Syms>(expr, seeds, grads);
    harvest(c, root, grads);
    toggle(c, S{});
  }
}

} // namespace detail

// The value-array APIs take CANONICAL (alphabetical) order; make_values() binds
// by name, so a missing, extra, duplicated or misspelled symbol will not
// compile.
template <CExpression Expr>
[[nodiscard]] consteval auto symbol_order() noexcept {
  using SymList = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  constexpr std::size_t N = mp::mp_size<SymList>::value;
  return index_apply<N>([]<std::size_t... I>() {
    return std::array<std::string_view, N>{mp::mp_at_c<SymList, I>::name...};
  });
}

// Scalar defaults to what derivative_tensor expects;
template <CExpression Expr,
          Numeric Scalar =
              scalar_base_t<typename std::remove_cvref_t<Expr>::value_type>,
          FixedString... Syms, Numeric... Vs>
[[nodiscard]] constexpr std::array<Scalar, detail::expr_arity_v<Expr>>
make_values(Entry<Syms, Vs>... nv) noexcept {
  using SymList = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  constexpr std::size_t N = mp::mp_size<SymList>::value;
  static_assert(sizeof...(Syms) == N,
                "make_values: supply exactly one value per symbol");
  static_assert(
      mp::mp_size<mp::mp_unique<mp::mp_list<symbol_type<Syms>...>>>::value ==
          sizeof...(Syms),
      "make_values: duplicate symbol");
  std::array<Scalar, N> out{};
  (
      [&]<FixedString Sy, Numeric Vv>(const Entry<Sy, Vv> &v) {
        constexpr std::size_t idx = symbol_index<Sy, SymList>();
        static_assert(idx < N, "make_values: symbol not present in expression");
        out[idx] = static_cast<Scalar>(v.value);
      }(nv),
      ...);
  return out;
}

namespace detail {

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

// One colour's harvest: (row, column, tangent) for every row whose colour owns
// a column; at most one per (colour, row), so that entry IS the sum.
constexpr void scatter(const auto &targets, const auto &grads,
                       auto &&sink) noexcept {
  for (const auto [i, target, g] :
       std::views::zip(std::views::iota(0uz), targets, grads)) {
    if (target != no_column) {
      sink(i, target, g.template get<1>());
    }
  }
}

// `Given` names the symbol list `values` is laid out by, as reverse_sweep's
// first parameter does; void is the expression's own.
template <typename Given = void, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = dual_scalar_t<T>,
          std::size_t N = mp::mp_size<sweep_symbols_t<Given, Expr>>::value>
  requires DualLike<T>
[[nodiscard]] constexpr auto
reverse_mode_hessian(const Expr &expr,
                     const std::array<S, N> &values) noexcept {
  using E = std::remove_cvref_t<Expr>;
  using Syms = sweep_symbols_t<Given, Expr>;
  static_assert(N == mp::mp_size<Syms>::value,
                "reverse_mode_hessian: one value per symbol of the list swept");

  // Sparsity is a property of the type, so the sweep loop carries no search.
  static constexpr const auto &kPattern = hessian_pattern_v<E, Syms>;
  static constexpr const auto &kColors = hessian_colors_v<E, N, Syms>;
  static constexpr auto kScatter = scatter_targets<N>(kPattern, kColors);

  // Value-initialised: the scatter writes only the pattern.  Symmetric-packed,
  // so (i, j) and (j, i) are one cell.
  nd_tensor_t<S, N, 2> H{};

  detail::color_sweeps<kColors, Syms>(
      expr, values, [&](std::size_t c, const T &, const auto &grads) {
        scatter(kScatter[c], grads,
                [&](std::size_t i, std::size_t j, const S &v) { H[i, j] = v; });
      });
  return H;
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

template <std::size_t Order, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = scalar_base_t<T>,
          std::size_t N = detail::expr_arity_v<Expr>>
  requires(Order > 0 && N > 0)
[[nodiscard]] constexpr auto
derivative_tensor_impl(const Expr &expr,
                       const std::array<S, N> &values) noexcept {
  using symbols = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  using U = nth_dual_t<S, Order>;

  nd_tensor_t<S, N, Order> result{};

  // J is a template parameter, so `k == J` folds away and the passes are
  // straight-line.
  if constexpr (Order == 1) {
    static_for<N>([&]<std::size_t Seeded>() {
      std::array<U, N> seeds{};
      for (const auto [k, v] : std::views::enumerate(values)) {
        const auto i = static_cast<std::size_t>(k);
        seeds[i] = U{v, i == Seeded ? S{1} : S{}};
      }
      result[Seeded] =
          expr.template eval_seeded<symbols>(seeds).template get<1>();
    });
  } else {
    for (const auto &idx : detail::simplex_index_table_v<N, Order>) {
      const auto seeds = mixed_seeds<S, Order>(values, idx);
      const U val = expr.template eval_seeded<symbols>(seeds);
      result.at_index(idx) = extract_nth<Order>(val);
    }
  }
  return result;
}

} // namespace detail

namespace detail {

// One Taylor sweep, then undo the 1/Order! normalisation.  Ref: Griewank,
// Utke & Walther, "Evaluating Higher Derivative Tensors by Forward Propagation
// of Univariate Taylor Series", Math. Comp. 69(231) (2000) 1117.
template <std::size_t Order, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = scalar_base_t<T>,
          std::size_t NVars = detail::expr_arity_v<Expr>>
  requires(Order > 0 && NVars == 1)
[[nodiscard]] DDX_ALWAYS_INLINE constexpr S
univariate_derivative_impl(const Expr &expr, S x0) noexcept {
  using symbols = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  using TD = TaylorDual<S, Order>;

  const TD result =
      expr.template eval_seeded<symbols>(std::array<TD, 1>{TD::variable(x0)});

  constexpr S factorial = static_cast<S>(compile_time_factorial(Order));
  return result.c[Order] * factorial;
}

} // namespace detail

namespace detail {

// Seeding column j and running one backward sweep yields that whole Hessian
// column: N sweeps against the scalar driver's N(N+1)/2 probes, with value and
// first derivatives free from the j == 0 sweep.
template <CExpression Expr>
constexpr HessianStatic<
    mp::mp_size<detail::expr_symbols_t<std::remove_cvref_t<Expr>>>::value>
hessian_expr_reverse(const Expr &expr, std::span<const double> x) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using Syms = detail::expr_symbols_t<E>;
  constexpr std::size_t N = mp::mp_size<Syms>::value;

  // A dense Hessian colours in N, so this degenerates to one sweep per column,
  // never worse.
  static constexpr const auto &kPattern = hessian_pattern_v<E>;
  static constexpr const auto &kColors = hessian_colors_v<E, N>;
  // Value-initialised is load-bearing: the scatter writes only the pattern.
  HessianStatic<N> res{};
  auto &res_jacobian = res.jacobian;
  const auto res_hessian = res.hessian_view();

  color_sweeps<kColors>(
      expr, x, [&](std::size_t c, const T &root, const auto &grads) {
        static constexpr auto kScatter = scatter_targets<N>(kPattern, kColors);
        if (c == 0) {
          // Neither depends on the tangent seeding, so any sweep yields both.
          res.value = static_cast<double>(root.template get<0>());
          std::ranges::transform(grads, res_jacobian.begin(), [](const T &g) {
            return static_cast<double>(g.template get<0>());
          });
        }

        scatter(kScatter[c], grads,
                [&](std::size_t i, std::size_t j, const auto &v) {
                  res_hessian[i, j] = static_cast<double>(v);
                });
      });

  // Independent sweeps, so mirrored entries can differ in the last ULP.
  detail::symmetrize(res_hessian);
  return res;
}

// The same sweeps, scattered straight into compressed storage.  The layout
// names each entry once, so nothing is symmetrised, and NNZ is a consteval
// count, so this path does not allocate.
template <CExpression Expr,
          std::size_t NNZ = hessian_nnz<std::remove_cvref_t<Expr>>()>
constexpr std::array<double, NNZ + 1>
hessian_values_sparse(const Expr &expr, std::span<const double> x) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using Syms = detail::expr_symbols_t<E>;
  constexpr std::size_t N = mp::mp_size<Syms>::value;

  static constexpr const auto &kColors = hessian_colors_v<E, N>;

  // One past the nonzeros: the sink every structural zero maps onto.
  std::array<double, NNZ + 1> values{};

  color_sweeps<kColors>(
      expr, x, [&](std::size_t c, const T &, const auto &grads) {
        static constexpr auto kSlots = sparse_slots<E>();
        scatter(kSlots[c], grads,
                [&](std::size_t, std::size_t slot, const auto &v) {
                  values[slot] = static_cast<double>(v);
                });
      });
  return values;
}

} // namespace detail

// The structure is in the expression type, so `outer` and `inner` are static
// constexpr and only the nnz values are computed: a CSC triple.
template <CExpression Expr> class SparseHessian {
  using E = std::remove_cvref_t<Expr>;
  static constexpr std::size_t kN = detail::expr_arity_v<E>;
  static constexpr auto kLayout = sparse_layout<E>();
  static constexpr std::size_t kNnz = decltype(kLayout)::nnz;

  // One past nnz: the sink cell, so reading an absent (i, j) is a load.
  std::array<double, kNnz + 1> values_;

public:
  static constexpr std::size_t rows = kN;
  static constexpr std::size_t nnz = kNnz;

  explicit constexpr SparseHessian(
      const std::array<double, kNnz + 1> &values) noexcept
      : values_(values) {}

  // Column j occupies [outer()[j], outer()[j + 1]); inner()[k] is the row of
  // stored value k.
  [[nodiscard]] static constexpr std::span<const int> outer() noexcept {
    return kLayout.outer;
  }
  [[nodiscard]] static constexpr std::span<const int> inner() noexcept {
    return kLayout.inner;
  }
  [[nodiscard]] constexpr std::span<const double> values() const & noexcept {
    return std::span<const double>{values_}.first(nnz);
  }
  auto values() const && = delete;

  [[nodiscard]] constexpr auto view() const & noexcept {
    return sparse_matrix_view<E>(values_);
  }
  auto view() const && = delete;
  [[nodiscard]] constexpr double operator[](std::size_t i,
                                            std::size_t j) const noexcept {
    return view()[i, j];
  }
  // In the pattern, as opposed to reading 0.0 because it cannot be otherwise.
  [[nodiscard]] static constexpr bool structural(std::size_t i,
                                                 std::size_t j) noexcept {
    return typename layout_sparse_pattern<E>::template mapping<
               md::extents<std::size_t, kN, kN>>{}
        .contains(i, j);
  }
};

// The sparse counterpart of hessian(graph, x).
[[nodiscard]] constexpr auto sparse_hessian(const CExpression auto &expr,
                                            std::span<const double> x) {
  return SparseHessian<std::remove_cvref_t<decltype(expr)>>{
      detail::hessian_values_sparse(expr, x)};
}

} // namespace ddx::impl
