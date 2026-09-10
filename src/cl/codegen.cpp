#include "codegen.hpp"

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
void expression(std::string &out, const rt::Graph<double> &g,
                std::size_t symbols, rt::NodeId v) {
  const auto &p = g[v];
  const auto [a, b, c] = g.operands(v);
  const std::string_view label = rt::label_of(p.op);
  auto sink = std::back_inserter(out);
  switch (rt::arity_of(p.op)) {
  case 0:
    if (p.op == rt::OpCode::Const) {
      out += literal(p.value);
    } else {
      std::format_to(sink, "xs[{}ul * n + i]",
                     rt::input_column(symbols, p.op, p.slot));
    }
    return;
  case 1:
    switch (p.op) {
    case rt::OpCode::Neg:
      std::format_to(sink, "-v{}", a);
      return;
    case rt::OpCode::Abs:
      std::format_to(sink, "fabs(v{})", a);
      return;
    // The last arm reaches only ±0 and NaN, as in the LLVM emitter.
    case rt::OpCode::Sign:
      std::format_to(sink, "(v{0} > 0.0 ? 1.0 : (v{0} < 0.0 ? -1.0 : v{0} - v{0}))", a);
      return;
    default:
      std::format_to(sink, "{}(v{})", label, a);
      return;
    }
  case 2:
    switch (p.op) {
    case rt::OpCode::Add:
    case rt::OpCode::Mul:
    case rt::OpCode::Div:
      std::format_to(sink, "v{} {} v{}", a, label, b);
      return;
    // 1.0 or 0.0, ordered: a NaN operand compares false.
    case rt::OpCode::Lt:
    case rt::OpCode::Le:
      std::format_to(sink, "(double)(v{} {} v{})", a, label, b);
      return;
    case rt::OpCode::Max:
    case rt::OpCode::Min:
      std::format_to(sink, "ddx_{}(v{}, v{})", label, a, b);
      return;
    default:
      std::format_to(sink, "{}(v{}, v{})", label, a, b);
      return;
    }
  default:
    // Select, the one ternary.  C's != is unordered, so a NaN condition takes
    // the first arm, as select_impl does.
    std::format_to(sink, "(v{} != 0.0 ? v{} : v{})", a, b, c);
    return;
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
  const std::size_t symbols = g.symbols().size();
  for (const auto &[v, fma] : schedule) {
    std::format_to(sink, "  const double v{} = ", v);
    if (fma) {
      std::format_to(sink, "fma({}v{}, v{}, v{})", fma.negated ? "-" : "",
                     fma.x, fma.y, fma.z);
    } else {
      expression(out, g, symbols, v);
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
