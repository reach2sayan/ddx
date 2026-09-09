#pragma once

#include "jit/kernel.hpp" // jit::Options -- a header type in every build
#include "rt/builder.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"

#include <boost/describe.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// What a built equation *is*, apart from how it is stored: the arena and the
// sweeps, not a frozen Graph, which freeze() rebuilds in one pass.  archive.hpp
// puts one of these on disk and takes it back; nothing here knows about bytes.
namespace ddx::rt {

// One lane's machine code, and what must agree before it runs; a mismatch means
// recompile, never an error.
struct Object {
  Want want = Want::Value;
  std::string symbol; // underivable, so adopt() is handed it back verbatim
  std::string host; // triple, CPU, folded features, LLVM version
  std::uint64_t digest = 0; // digest() of the lane's frozen graph
  jit::Codegen codegen;     // what the compile was given
  std::vector<std::byte> code;
  BOOST_DESCRIBE_CLASS(Object, (), (want, symbol, host, digest, codegen, code),
                       (), ())
};

// The stored kernel for a lane, if graph, host and codegen options all agree:
// adopt() cannot see that an object came from another graph, so nothing
// reaches it unchecked.
[[nodiscard]] inline const Object *find_object(std::span<const Object> objects,
                                               Want want, std::uint64_t digest,
                                               std::string_view host,
                                               const jit::Codegen &codegen) {
  const auto it = std::ranges::find_if(objects, [&](const Object &o) {
    return o.want == want && o.digest == digest && o.host == host &&
           o.codegen == codegen && !o.code.empty();
  });
  return it == objects.end() ? nullptr : &*it;
}

// All the C++ facade and PyEquation share, and so the only thing serialised.
template <impl::Numeric T> struct Snapshot {
  std::vector<std::string> symbols;
  std::vector<Node<T>> nodes;
  std::vector<NodeId> roots;
  Jacobian jacobian;
  std::vector<Hessian> hessians;
  // Serialised rather than re-swept: Equation(Snapshot&&) calls no build_*, so
  // the file describes every block its lanes can freeze.
  HessianVector hvp;
  VectorJacobian vjp;
  Tangent jvp;
  jit::Options options;
  // The staleness key: rebuilding the model reproduces this prefix.
  std::uint32_t model_nodes = 0;
  std::vector<Object> objects;
  BOOST_DESCRIBE_CLASS(Snapshot, (),
                       (symbols, nodes, roots, jacobian, hessians, hvp, vjp,
                        jvp, options, model_nodes, objects),
                       (), ())
};

} // namespace ddx::rt
