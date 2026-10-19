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
// writes by hand.  The one list the LLVM emitter and the runtime's symbol
// table read; a leaf and an extern have no spelling here.
namespace ddx::rt {

enum class Form : std::uint8_t { None, Infix, Compare, Call, Special };
struct Spelling {
  Form form = Form::None;
  std::string_view name{};
};

namespace detail {
inline constexpr std::array spellings = [] {
  std::array<Spelling, op_count> t{};
  const auto set = [&t](OpCode op, Form form, std::string_view name = {}) {
    t[std::to_underlying(op)] = {form, name.empty() ? label_of(op) : name};
  };
  for (const OpCode op : {OpCode::Add, OpCode::Mul, OpCode::Div}) {
    set(op, Form::Infix);
  }
  for (const OpCode op : {OpCode::Lt, OpCode::Le}) {
    set(op, Form::Compare);
  }
  set(OpCode::Abs, Form::Call, "fabs"); // abs is the integer one in C
  for (const OpCode op : {OpCode::Pow, OpCode::Atan2, OpCode::Hypot}) {
    set(op, Form::Call);
  }
#define DDX_RT_SPELL(fn, Op, label) set(OpCode::Op, Form::Call);
  DDX_UNARY_MATH_TABLE(DDX_RT_SPELL)
#undef DDX_RT_SPELL
  for (const OpCode op :
       {OpCode::Neg, OpCode::Sign, OpCode::Max, OpCode::Min, OpCode::Select}) {
    set(op, Form::Special);
  }
  return t;
}();
} // namespace detail

[[nodiscard]] constexpr const Spelling &spelling_of(OpCode op) noexcept {
  return detail::spellings[std::to_underlying(op)];
}

// Every operation a backend may meet has a spelling.
static_assert(std::ranges::all_of(std::views::iota(0uz, op_count),
                                  [](std::size_t i) {
                                    const auto op = static_cast<OpCode>(i);
                                    return !callable(op) ||
                                           spelling_of(op).form != Form::None;
                                  }));

} // namespace ddx::rt
