#pragma once

#include <boost/config.hpp>

// Outside MSVC both sides use default visibility, so a consumer sees the
// instances as exported rather than hidden.
#if defined(_MSC_VER) && !defined(DDX_BUILDING)
#define DDX_API BOOST_SYMBOL_IMPORT
#else
#define DDX_API BOOST_SYMBOL_EXPORT
#endif

// For templates that libddx instantiates explicitly. GCC and Clang need the
// attribute on the template, MSVC on the instantiation line.
#if defined(_MSC_VER)
#define DDX_TEMPLATE_API
#define DDX_INSTANCE_API DDX_API
#else
#define DDX_TEMPLATE_API DDX_API
#define DDX_INSTANCE_API
#endif

// Declares an instantiation that libddx makes, so no TU builds hidden copies
// that would win at link. MSVC rejects it inside libddx itself (C4910).
#if defined(_MSC_VER) && defined(DDX_BUILDING)
#define DDX_EXTERN_INSTANCE(...)
#elif defined(_MSC_VER)
#define DDX_EXTERN_INSTANCE(...)                                               \
  __pragma(warning(push))                                                      \
      __pragma(warning(disable : 4251)) extern template __VA_ARGS__;           \
  __pragma(warning(pop))
#else
#define DDX_EXTERN_INSTANCE(...) extern template __VA_ARGS__;
#endif
