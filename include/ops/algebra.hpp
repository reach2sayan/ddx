#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

// The algebraic identities shared by the type-level and graph simplifiers.
namespace ddx::impl::algebra {

enum class RuleOp : std::uint8_t { Add, Mul, Div, Pow, Neg };

// Conditions that each simplifier tests in its own terms, by type trait or by
// node id.
enum class Pred : std::uint8_t {
  ZeroA,   // a is the literal 0
  ZeroB,   // b is the literal 0
  OneA,    // a is the literal 1
  OneB,    // b is the literal 1
  Same,    // a and b are the same expression
  TwoB,    // b is the literal 2
  AOverB,  // a is a quotient whose denominator is exactly b
  BOverA,  // b is a quotient whose denominator is exactly a
  NegatedA // a is itself a negation
};

// What a matched node is replaced by.
enum class Take : std::uint8_t {
  LitZero,
  LitOne,
  OperandA,
  OperandB,
  NumeratorOfA, // a is n/b; take n
  NumeratorOfB, // b is n/a; take n
  OperandOfA,   // a is -x; take x
  SquareOfA     // a * a
};

struct Rule {
  RuleOp op;
  Pred when;
  Take then;
  // a * (n/a) is a*n*a^-1, which is n only when the factors commute. (n/a) * a
  // needs no guard, because / is right division.
  bool needs_commutative_multiply = false;
};

// Order matters, since both simplifiers take the first match.
inline constexpr std::array kRules{
    // clang-format off
    Rule{.op = RuleOp::Mul, .when = Pred::ZeroA,  .then = Take::LitZero},
    Rule{.op = RuleOp::Mul, .when = Pred::ZeroB,  .then = Take::LitZero},
    Rule{.op = RuleOp::Mul, .when = Pred::OneA,   .then = Take::OperandB},
    Rule{.op = RuleOp::Mul, .when = Pred::OneB,   .then = Take::OperandA},
    Rule{.op = RuleOp::Mul, .when = Pred::AOverB, .then = Take::NumeratorOfA},
    Rule{.op = RuleOp::Mul, .when = Pred::BOverA, .then = Take::NumeratorOfB, .needs_commutative_multiply = true},

    Rule{.op = RuleOp::Add, .when = Pred::ZeroA,  .then = Take::OperandB},
    Rule{.op = RuleOp::Add, .when = Pred::ZeroB,  .then = Take::OperandA},

    Rule{.op = RuleOp::Div, .when = Pred::Same,   .then = Take::LitOne},
    Rule{.op = RuleOp::Div, .when = Pred::ZeroA,  .then = Take::LitZero},
    Rule{.op = RuleOp::Div, .when = Pred::OneB,   .then = Take::OperandA},

    Rule{.op = RuleOp::Pow, .when = Pred::ZeroB,  .then = Take::LitOne},
    Rule{.op = RuleOp::Pow, .when = Pred::OneB,   .then = Take::OperandA},
    // x^2 is the one integer power that a single multiply gives exactly. The
    // reverse rule creates pow(u, r-1) for every power, so this one pays.
    Rule{.op = RuleOp::Pow, .when = Pred::TwoB,     .then = Take::SquareOfA},

    Rule{.op = RuleOp::Neg, .when = Pred::NegatedA, .then = Take::OperandOfA},
    // clang-format on
};

// The operand count of an op. No default, so a new enumerator must be added.
[[nodiscard]] constexpr std::size_t arity_of(RuleOp op) noexcept {
  switch (op) {
  case RuleOp::Add:
  case RuleOp::Mul:
  case RuleOp::Div:
  case RuleOp::Pow:
    return 2;
  case RuleOp::Neg:
    return 1;
  }
  std::unreachable();
}

// The operands a predicate reads; the *A predicates read only a.
[[nodiscard]] constexpr std::size_t arity_of(Pred p) noexcept {
  switch (p) {
  case Pred::ZeroA:
  case Pred::OneA:
  case Pred::NegatedA:
    return 1;
  case Pred::ZeroB:
  case Pred::OneB:
  case Pred::Same:
  case Pred::TwoB:
  case Pred::AOverB:
  case Pred::BOverA:
    return 2;
  }
  std::unreachable();
}

static_assert(std::ranges::all_of(kRules,
                                  [](const Rule &r) {
                                    return arity_of(r.when) <= arity_of(r.op);
                                  }),
              "a rule reads no operand its op lacks");

} // namespace ddx::impl::algebra
