#pragma once

#include "rt/opcode.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ranges>
#include <string_view>
#include <utility>

// How a backend spells each operation: an infix operator, an ordered compare
// answering 1.0 or 0.0, a call by name -- an intrinsic where the backend has
// one, else a libm function the runtime defines -- or a form the backend
// writes by hand.  The one list the LLVM emitter, the OpenCL emitter and the
// runtime's symbol table read; a leaf and an extern have no spelling here.
namespace ddx::rt {

enum class Lower : std::uint8_t { None, Infix, Compare, Call, Special };
struct Lowering {
  Lower how = Lower::None;
  std::string_view name{};
};

namespace detail {
inline constexpr std::array lowering = [] {
  std::array<Lowering, op_count> t{};
  const auto set = [&t](OpCode op, Lower how, std::string_view name = {}) {
    t[std::to_underlying(op)] = {how, name.empty() ? label_of(op) : name};
  };
  for (const OpCode op : {OpCode::Add, OpCode::Mul, OpCode::Div}) {
    set(op, Lower::Infix);
  }
  for (const OpCode op : {OpCode::Lt, OpCode::Le}) {
    set(op, Lower::Compare);
  }
  set(OpCode::Abs, Lower::Call, "fabs"); // abs is the integer one in C
  for (const OpCode op : {OpCode::Pow, OpCode::Atan2, OpCode::Hypot}) {
    set(op, Lower::Call);
  }
#define DDX_RT_LOWER(fn, Op, label) set(OpCode::Op, Lower::Call);
  DDX_UNARY_MATH_TABLE(DDX_RT_LOWER)
#undef DDX_RT_LOWER
  for (const OpCode op :
       {OpCode::Neg, OpCode::Sign, OpCode::Max, OpCode::Min, OpCode::Select}) {
    set(op, Lower::Special);
  }
  return t;
}();
} // namespace detail

[[nodiscard]] constexpr const Lowering &lowering_of(OpCode op) noexcept {
  return detail::lowering[std::to_underlying(op)];
}

// Every operation a backend may meet has a spelling.
static_assert(std::ranges::all_of(std::views::iota(0uz, op_count),
                                  [](std::size_t i) {
                                    const auto op = static_cast<OpCode>(i);
                                    return !callable(op) ||
                                           lowering_of(op).how != Lower::None;
                                  }));

} // namespace ddx::rt
