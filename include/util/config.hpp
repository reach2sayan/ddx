#pragma once
#include <utility>

#define DDX_FWD(X) std::forward<decltype(X)>(X)

// Every value category *this can arrive as, with the expression that forwards
// it: X(qualifier, self).
#define DDX_VALUE_CATEGORIES(X)                                                \
  X(&, (*this))                                                                \
  X(const &, (*this))                                                          \
  X(&&, std::move(*this))                                                      \
  X(const &&, std::move(*this))

// The sweep helpers and dual kernels are factored-out code, not calls; GCC
// stops inlining them once a TU exhausts its inlining budget.
#if defined(__GNUC__) || defined(__clang__)
#define DDX_ALWAYS_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
#define DDX_ALWAYS_INLINE __forceinline
#else
#define DDX_ALWAYS_INLINE inline
#endif

// What the JIT spells NoAlias on the kernel's columns.  A tape reached through
// a span carries no such promise, and without it every lane store is assumed
// to land in the node array the next lane load reads.
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define DDX_RESTRICT __restrict
#else
#define DDX_RESTRICT
#endif

// Accessors for a class with a private static slot(auto &&self).  ... is a
// trailing requires-clause.  The four categories are spelled out rather than
// run through DDX_VALUE_CATEGORIES: MSVC's default preprocessor does not
// re-split a forwarded __VA_ARGS__, so only the tail may be forwarded.
#define DDX_SLOT_OVERLOAD_(Q, SELF, HEAD, NAME, PARAM, KEY, ...)               \
  HEAD [[nodiscard]] constexpr decltype(auto) NAME(PARAM)                      \
      Q noexcept __VA_ARGS__ {                                                 \
    return slot<KEY>(SELF);                                                    \
  }
#define DDX_SLOT_OVERLOADS_(HEAD, NAME, PARAM, KEY, ...)                       \
  DDX_SLOT_OVERLOAD_(&, *this, HEAD, NAME, PARAM, KEY, __VA_ARGS__)            \
  DDX_SLOT_OVERLOAD_(const &, *this, HEAD, NAME, PARAM, KEY, __VA_ARGS__)      \
  DDX_SLOT_OVERLOAD_(&&, std::move(*this), HEAD, NAME, PARAM, KEY,             \
                     __VA_ARGS__)                                              \
  DDX_SLOT_OVERLOAD_(const &&, std::move(*this), HEAD, NAME, PARAM, KEY,       \
                     __VA_ARGS__)

// One slot under a name of its own: no key parameter, the name is the key.
#define DDX_SLOT_ACCESSOR(NAME, KEY) DDX_SLOT_OVERLOADS_(, NAME, , KEY, )
#define DDX_KEYED_GET(TPARAMS, KEY, ...)                                       \
  DDX_SLOT_OVERLOADS_(template <TPARAMS>, get, , KEY, __VA_ARGS__)
// The same slot, reached through the empty tag operator[] deduces its key from.
#define DDX_KEYED_SUBSCRIPT(TPARAMS, KEY, SUB_PARAM, ...)                      \
  DDX_SLOT_OVERLOADS_(template <TPARAMS>, operator[], SUB_PARAM, KEY,          \
                      __VA_ARGS__)
// Both spellings of one slot.  The two parameter lists differ because get<>
// takes its key and operator[] deduces it from the tag.
#define DDX_KEYED_ACCESSORS(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, ...)     \
  DDX_KEYED_GET(GET_TPARAMS, KEY, __VA_ARGS__)                                 \
  DDX_KEYED_SUBSCRIPT(SUB_TPARAMS, KEY, SUB_PARAM, __VA_ARGS__)
