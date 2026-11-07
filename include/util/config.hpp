#pragma once
#include <boost/config.hpp>

#include <utility>

#define DDX_FWD(X) std::forward<decltype(X)>(X)

// DDX_TRY(x, f()); binds x to f()'s expected or returns its error. DDX_CHECK
// does the same for an expected whose value is not needed.
#define DDX_TRY(var, expr)                                                     \
  auto var = (expr);                                                           \
  if (!var) [[unlikely]]                                                       \
    return std::unexpected { std::move(var).error() }
#define DDX_CHECK(expr)                                                        \
  if (auto ddx_checked_ = (expr); !ddx_checked_) [[unlikely]]                  \
    return std::unexpected { std::move(ddx_checked_).error() }

// For sweep helpers and dual kernels, which GCC stops inlining once a TU
// exhausts its inlining budget.
#define DDX_ALWAYS_INLINE BOOST_FORCEINLINE
// For a lane loop that GCC vectorises on its own but not once inlined.
#define DDX_NOINLINE BOOST_NOINLINE
// The JIT's NoAlias for tape columns. Without it a lane store through a span is
// assumed to alias the node array that the next lane load reads.
#define DDX_RESTRICT BOOST_RESTRICT
