#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace ddx::impl {

// Sets slot to next and restores the old value on scope exit. Not movable, or
// a second destructor would write back a moved-from value.
template <typename T>
  requires std::move_constructible<T> && std::assignable_from<T &, T>
class scoped_value {
public:
  constexpr scoped_value(T &slot, T next) noexcept(
      std::is_nothrow_move_constructible_v<T> &&
      std::is_nothrow_move_assignable_v<T>)
      : slot_(slot), saved_(std::exchange(slot, std::move(next))) {}
  scoped_value(const scoped_value &) = delete;
  scoped_value &operator=(const scoped_value &) = delete;
  constexpr ~scoped_value() { slot_ = std::move(saved_); }

private:
  T &slot_;
  T saved_;
};

} // namespace ddx::impl
