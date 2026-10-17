#pragma once

#include "jit/options.hpp"
#include "rt/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// The object cache on disk: one entry per (graph, host, codegen) key, holding
// the symbol and the object bytes a compile produced.  Content-addressed, so a
// stale or damaged entry is a miss, and nothing here fails a compile.
namespace ddx::jit::detail {

struct Entry {
  std::string symbol;
  std::vector<std::byte> code;
};

// Keyed on what codegen reads of a lane: the layout, every scheduled node's
// property and operands, and the outputs.  Ids are construction-ordered, so
// this is a within-one-binary key.
[[nodiscard]] std::uint64_t digest(const rt::Graph<double> &g);

[[nodiscard]] std::uint64_t
cache_key(std::uint64_t graph, std::string_view host, const Codegen &emitted);

[[nodiscard]] std::filesystem::path entry_path(std::string_view dir,
                                               std::uint64_t key);

[[nodiscard]] std::optional<Entry> read_entry(std::string_view dir,
                                              std::uint64_t key);

void write_entry(std::string_view dir, std::uint64_t key,
                 std::string_view symbol, std::span<const std::byte> code);

} // namespace ddx::jit::detail
