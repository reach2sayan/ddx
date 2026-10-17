#include "codegen.hpp"

#include "rt/lowering.hpp"

#include "cl/kernel.hpp"
#include "rt/opcode.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <iterator>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

namespace ddx::cl::detail {
namespace {

// extremum_impl<IsMax> in ops/operations.hpp, transliterated: a tie answers
// `a` except between two zeros, where the sum (max) or the negated sum of the
// negations (min) puts -0 below +0; an unordered pair answers (a - b) * 0, NaN
// from either side.
constexpr std::string_view extrema = R"(
double ddx_max(double a, double b) {
  if (a == b) return a == 0.0 ? a + b : a;
  if (a < b) return b;
  if (b < a) return a;
  return (a - b) * 0.0;
}
double ddx_min(double a, double b) {
  if (a == b) return a == 0.0 ? -((-a) + (-b)) : a;
  if (a < b) return a;
  if (b < a) return b;
  return (a - b) * 0.0;
}
)";

// Hex, so the literal is the value's own bits.  std::format's `a` leaves off
// the 0x OpenCL C wants, and a non-finite value goes through its bits because
// OpenCL's NAN and INFINITY are float.
[[nodiscard]] std::string literal(double v) {
  if (!std::isfinite(v)) {
    return std::format("as_double(0x{:016x}ul)", std::bit_cast<std::uint64_t>(v));
  }
  return std::format("{}0x{:a}", std::signbit(v) ? "-" : "", std::fabs(v));
}

// One node's right-hand side, mirroring the LLVM emitter's rules node for node.
void expression(std::string &out, const rt::Graph<double> &g, rt::NodeId v) {
  const auto &p = g[v];
  const auto [a, b, c] = g.operands(v);
  const std::string_view label = rt::label_of(p.op);
  auto sink = std::back_inserter(out);
  switch (rt::kind_of(p.op)) {
  case rt::Kind::Const:
    out += literal(p.value);
    return;
  case rt::Kind::Input:
    std::format_to(sink, "xs[{}ul * n + i]",
                   rt::input_column(g.inputs(), p.op, p.slot));
    return;
  case rt::Kind::Nary:
    std::unreachable(); // a statement per term: sum() below
  case rt::Kind::Extern:
  case rt::Kind::Tensor:
    std::unreachable(); // Device::compile refuses a graph that calls the host
                        // or holds a tensor row
  case rt::Kind::Unary:
  case rt::Kind::Binary: {
    const auto &[how, name] = rt::lowering_of(p.op);
    switch (how) {
    case rt::Lower::Infix:
      std::format_to(sink, "v{} {} v{}", a, name, b);
      return;
    // 1.0 or 0.0, ordered: a NaN operand compares false.
    case rt::Lower::Compare:
      std::format_to(sink, "(double)(v{} {} v{})", a, name, b);
      return;
    case rt::Lower::Call:
      if (rt::arity_of(p.op) == 1) {
        std::format_to(sink, "{}(v{})", name, a);
      } else {
        std::format_to(sink, "{}(v{}, v{})", name, a, b);
      }
      return;
    case rt::Lower::Special:
    case rt::Lower::None:
      break;
    }
    switch (p.op) {
    case rt::OpCode::Neg:
      std::format_to(sink, "-v{}", a);
      return;
    // The last arm reaches only ±0 and NaN, as in the LLVM emitter.
    case rt::OpCode::Sign:
      std::format_to(sink, "(v{0} > 0.0 ? 1.0 : (v{0} < 0.0 ? -1.0 : v{0} - v{0}))", a);
      return;
    default: // Max and Min, through the prelude's NaN-propagating pair
      std::format_to(sink, "ddx_{}(v{}, v{})", label, a, b);
      return;
    }
  }
  case rt::Kind::Ternary:
    // Select, the one ternary.  C's != is unordered, so a NaN condition takes
    // the first arm, as select_impl does.
    std::format_to(sink, "(v{} != 0.0 ? v{} : v{})", a, b, c);
    return;
  }
}

// A sum, one statement per term so no expression nests: the terms into the
// running sum in list order, as the LLVM emitter and the sweep take them -- a
// product term one fma() where the step is fused, the first term formed.
void sum(std::string &out, const rt::Graph<double> &g, const rt::Step &s) {
  auto sink = std::back_inserter(out);
  const rt::NodeId v = s.node;
  for (const auto [i, t] : rt::arguments(g, v) | std::views::enumerate) {
    const rt::Contraction f =
        s.fused ? rt::product_of(g, t) : rt::Contraction{};
    const std::string_view sign = f.negated ? "-" : "";
    if (i == 0) {
      if (f) {
        std::format_to(sink, "  double v{} = {}(v{} * v{});\n", v, sign, f.x, f.y);
      } else {
        std::format_to(sink, "  double v{} = v{};\n", v, t);
      }
    } else if (f) {
      std::format_to(sink, "  v{0} = fma({1}v{2}, v{3}, v{0});\n", v, sign, f.x, f.y);
    } else {
      std::format_to(sink, "  v{} += v{};\n", v, t);
    }
  }
}

} // namespace

std::string emit_source(const rt::Graph<double> &g) {
  std::string out;
  auto sink = std::back_inserter(out);
  // FP_CONTRACT OFF: contraction is decided in the graph and spelled fma(), so
  // a compiler fusing on its own would round where the sweep does not.
  out += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
         "#pragma OPENCL FP_CONTRACT OFF\n";
  const auto schedule = g.schedule();
  if (std::ranges::any_of(schedule, [&g](const rt::Step &s) {
        const rt::OpCode op = g.op_of(s.node);
        return !s.fma && (op == rt::OpCode::Max || op == rt::OpCode::Min);
      })) {
    out += extrema;
  }
  std::format_to(sink,
                 "\n__kernel void {}(__global const double *restrict xs,\n"
                 "                 __global double *restrict f,\n"
                 "                 __global double *restrict g,\n"
                 "                 __global double *restrict h,\n"
                 "                 const ulong n) {{\n"
                 "  const ulong i = get_global_id(0);\n"
                 "  if (i >= n) {{\n"
                 "    return;\n"
                 "  }}\n",
                 kernel_name);

  // Ids are topological, so one pass needs no worklist, and a multiply every
  // reader swallowed into an fma is never reached.
  for (const rt::Step &s : schedule) {
    const auto &[v, fma] = std::pair{s.node, s.fma};
    if (rt::kind_of(g.op_of(v)) == rt::Kind::Nary) {
      sum(out, g, s);
      continue;
    }
    std::format_to(sink, "  const double v{} = ", v);
    if (fma) {
      std::format_to(sink, "fma({}v{}, v{}, v{})", fma.negated ? "-" : "",
                     fma.x, fma.y, fma.z);
    } else {
      expression(out, g, v);
    }
    out += ";\n";
  }

  static constexpr std::array names{"f", "g", "h"};
  for (const auto [name, block] :
       std::views::zip(names, rt::in_order(g.output_blocks()))) {
    for (const auto [j, o] : std::views::enumerate(block)) {
      std::format_to(sink, "  {}[{}ul * n + i] = v{};\n", name, j, o);
    }
  }
  out += "}\n";
  return out;
}

} // namespace ddx::cl::detail

namespace ddx::cl {

std::string source_of(const rt::Graph<double> &g) {
  return detail::emit_source(g);
}

} // namespace ddx::cl
