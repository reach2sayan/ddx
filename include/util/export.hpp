#pragma once

#include <boost/config.hpp>

// Default visibility outside MSVC either way: a consumer's TU must see the
// instances as exported, not hidden.
#if defined(_MSC_VER) && !defined(DDX_BUILDING)
#define DDX_API BOOST_SYMBOL_IMPORT
#else
#define DDX_API BOOST_SYMBOL_EXPORT
#endif

#define DDX_JIT_API DDX_API

// A template libddx instantiates explicitly (DDX_RT_INSTANCES).  GCC and Clang
// take the visibility on the template, since an instantiation line refuses it
// for a type a TU completed earlier; MSVC takes it on the instantiation line,
// since on the template it would import every specialisation.
#if defined(_MSC_VER)
#define DDX_TEMPLATE_API
#define DDX_INSTANCE_API DDX_API
#else
#define DDX_TEMPLATE_API DDX_API
#define DDX_INSTANCE_API
#endif

// The extern half of an instantiation libddx makes once, written beside the
// class: a TU that sees the class and not this instantiates hidden copies of
// its inline members, and the linker keeps the hidden one.  MSVC refuses
// dllexport on it (C4910), so libddx's own TUs skip it there; C4251 is the
// std:: members of an exported class, and both sides share one STL.
#if defined(_MSC_VER) && defined(DDX_BUILDING)
#define DDX_EXTERN_INSTANCE(...)
#elif defined(_MSC_VER)
#define DDX_EXTERN_INSTANCE(...)                                               \
  __pragma(warning(push)) __pragma(warning(disable : 4251))                    \
      extern template __VA_ARGS__;                                             \
  __pragma(warning(pop))
#else
#define DDX_EXTERN_INSTANCE(...) extern template __VA_ARGS__;
#endif
