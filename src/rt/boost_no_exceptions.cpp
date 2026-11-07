// Under BOOST_NO_EXCEPTIONS the program must define boost::throw_exception.
// Only Boost internals reach it, and aborting is the only safe response.
//
// Visible by default because libddx is -fvisibility=hidden and Boost's inline
// code in other objects must find these. DDX_API is for ddx's own namespace.
#include <boost/assert/source_location.hpp>
#include <boost/throw_exception.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace boost {

BOOST_SYMBOL_VISIBLE void throw_exception(const std::exception &e) {
  std::fprintf(stderr, "ddx: boost threw with exceptions disabled: %s\n",
               e.what());
  std::abort();
}

BOOST_SYMBOL_VISIBLE void throw_exception(const std::exception &e,
                                          const source_location &loc) {
  std::fprintf(stderr,
               "ddx: boost threw with exceptions disabled: %s (%s:%u)\n",
               e.what(), loc.file_name(), loc.line());
  std::abort();
}

} // namespace boost
