#pragma once

#include "rt/opcode.hpp"
#include "util/error.hpp"
#include "util/export.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ddx::rt::text {

inline constexpr std::uint32_t no_term = ~std::uint32_t{0};

// Typed leaf indices, so a name index cannot be used to read the literals.
struct NameIndex {
  static constexpr OpCode op = OpCode::Var;
  std::uint32_t i = 0;
};
struct LiteralIndex {
  static constexpr OpCode op = OpCode::Const;
  std::uint32_t i = 0;
};
using Leaf = std::variant<std::monostate, NameIndex, LiteralIndex>;

// An untyped parsed node: an opcode, its operand indices (unused ones are
// no_term), and for leaves the index into the names or literals.
struct Term {
  OpCode op = OpCode::Const;
  std::array<std::uint32_t, 3> in{no_term, no_term, no_term};
  Leaf leaf{};
};

struct Ast {
  std::vector<Term> terms;
  std::vector<std::string> names;    // free identifiers in order of first use
  std::vector<std::string> literals; // kept as text so they are read at T
  std::uint32_t root = no_term;

  template <typename Index>
    requires std::same_as<Index, NameIndex> || std::same_as<Index, LiteralIndex>
  [[nodiscard]] std::vector<std::string> &table() noexcept {
    if constexpr (std::same_as<Index, NameIndex>) {
      return names;
    } else {
      return literals;
    }
  }
};

[[nodiscard]] DDX_API result<Ast> parse(std::string_view source);

} // namespace ddx::rt::text
