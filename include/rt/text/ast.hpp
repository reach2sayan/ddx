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

// A leaf's index carries which table it names, so a Var cannot be read against
// the literals.
struct NameIndex {
  static constexpr OpCode op = OpCode::Var;
  std::uint32_t i = 0;
};
struct LiteralIndex {
  static constexpr OpCode op = OpCode::Const;
  std::uint32_t i = 0;
};
using Leaf = std::variant<std::monostate, NameIndex, LiteralIndex>;

// A node before it has a value type: an opcode, the operands its arity names
// (the rest no_term), and -- for the two leaves -- the index that names it.
struct Term {
  OpCode op = OpCode::Const;
  std::array<std::uint32_t, 3> in{no_term, no_term, no_term};
  Leaf leaf{};
};

struct Ast {
  std::vector<Term> terms;
  std::vector<std::string> names;    // free identifiers, first named first
  std::vector<std::string> literals; // as written: read at T, never at double
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
