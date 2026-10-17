#pragma once

#include <boost/core/noncopyable.hpp>

namespace ddx::impl {

namespace pinned_ {

// One owner, and the ownership can be handed on.
class noncopyable {
protected:
  constexpr noncopyable() = default;
  ~noncopyable() = default;
  noncopyable(noncopyable &&) = default;
  noncopyable &operator=(noncopyable &&) = default;

public:
  noncopyable(const noncopyable &) = delete;
  noncopyable &operator=(const noncopyable &) = delete;
};

} // namespace pinned_

using pinned_::noncopyable;
// The same, and the ownership cannot be handed on either: Boost's declares
// the copy deleted and so has no move.
using pinned = boost::noncopyable;

} // namespace ddx::impl
