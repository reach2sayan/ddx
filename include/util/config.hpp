#pragma once
#include <boost/config.hpp>

#include <utility>

#define DDX_FWD(X) std::forward<decltype(X)>(X)

// The early return over std::expected, spelled `DDX_TRY(x, f());`: binds `x`
// to f()'s expected, or returns its error.  DDX_CHECK is the same for an
// expected whose value nothing reads.
#define DDX_TRY(var, expr)                                                     \
  auto var = (expr);                                                           \
  if (!var) [[unlikely]]                                                       \
    return std::unexpected{std::move(var).error()}
#define DDX_CHECK(expr)                                                        \
  if (auto ddx_checked_ = (expr); !ddx_checked_) [[unlikely]]                  \
    return std::unexpected{std::move(ddx_checked_).error()}

// The sweep helpers and dual kernels are factored-out code, not calls; GCC
// stops inlining them once a TU exhausts its inlining budget.
#define DDX_ALWAYS_INLINE BOOST_FORCEINLINE
// The converse: a lane loop GCC vectorises on its own and not once inlined.
#define DDX_NOINLINE BOOST_NOINLINE
// What the JIT spells NoAlias on the kernel's columns.  A tape reached through
// a span carries no such promise, and without it every lane store is assumed
// to land in the node array the next lane load reads.
#define DDX_RESTRICT BOOST_RESTRICT
