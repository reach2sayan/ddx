#pragma once

#include "ops/numeric.hpp"
#include "util/error.hpp"

#include <boost/describe.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>

// What a caller says about compiling, and nothing that compiles: header types
// in every build, so an Equation has one shape whether or not a backend was
// compiled in.  jit/kernel.hpp holds the LLVM side.
namespace ddx::rt {
template <impl::Numeric T> class Graph;
namespace detail {
template <impl::Numeric T> struct TensorCall;
} // namespace detail
} // namespace ddx::rt

namespace ddx::jit {

// Off: glibc's vector routines are ~4 ULP where the scalar ones are ~0.5.
enum class VecLib : std::uint8_t { None, Auto, Libmvec };
BOOST_DESCRIBE_ENUM(VecLib, None, Auto, Libmvec)

struct compile_error {
  errc code;
  std::string detail;
};

template <typename T> using compile_result = std::expected<T, compile_error>;

#ifndef DDX_JIT_DEFAULT_OPT
#define DDX_JIT_DEFAULT_OPT 2
#endif
#ifndef DDX_JIT_DEFAULT_CONTRACT
#define DDX_JIT_DEFAULT_CONTRACT 1
#endif

// LLVM's -O0 to -O3, for the IR pipeline and for codegen.
enum class OptLevel : std::uint8_t { O0, O1, O2, O3 };
BOOST_DESCRIBE_ENUM(OptLevel, O0, O1, O2, O3)

static_assert(DDX_JIT_DEFAULT_OPT >= 0 && DDX_JIT_DEFAULT_OPT <= 3);
inline constexpr OptLevel default_opt_level =
    static_cast<OptLevel>(DDX_JIT_DEFAULT_OPT);
inline constexpr bool default_contract = DDX_JIT_DEFAULT_CONTRACT != 0;

// Points per loop iteration.  Derived is the host's register width in doubles
// -- scalar for a batch too short to fill one, and under a vector library the
// widest it serves.  Every width gives the same bits.
class Lanes {
public:
  constexpr Lanes() = default;
  [[nodiscard]] static constexpr Lanes derived() noexcept { return {}; }
  [[nodiscard]] static constexpr Lanes scalar() noexcept { return Lanes{1}; }
  // A stated width holds at least a point; nullopt otherwise.
  [[nodiscard]] static constexpr std::optional<Lanes>
  exactly(unsigned width) noexcept {
    return width > 0 ? std::optional{Lanes{width}} : std::nullopt;
  }
  [[nodiscard]] constexpr std::optional<unsigned> stated() const noexcept {
    return width_ > 0 ? std::optional{width_} : std::nullopt;
  }
  friend constexpr bool operator==(Lanes, Lanes) = default;

private:
  constexpr explicit Lanes(unsigned width) noexcept : width_(width) {}
  unsigned width_ = 0;
  BOOST_DESCRIBE_CLASS(Lanes, (), (), (), (width_))
};

// Compile builds each kernel on the first call that needs it, and that call
// waits for it.
enum class Backend : std::uint8_t { Interpret, Compile };
BOOST_DESCRIBE_ENUM(Backend, Interpret, Compile)

// What decides the machine code, and nothing else.
struct Codegen {
  Lanes lanes = Lanes::derived();
  OptLevel opt_level = default_opt_level;
  // Selection and register allocation are ~95% of a compile.
  OptLevel codegen_level = OptLevel::O1;
  // Within one point; model-dependent, hence off.
  bool slp_vectorize = false;
  // Off: the body is already `lanes` wide, and past ~19 columns the alias
  // checks exceed their own budget.
  bool loop_vectorize = false;
  // With one, a derived `lanes` is the widest width the library serves -- four
  // doubles for libmvec -- whatever the host's registers hold.
  VecLib veclib = VecLib::None;
  bool contract = default_contract; // Follows DDX_FP_FLAGS

  friend bool operator==(const Codegen &, const Codegen &) = default;
  BOOST_DESCRIBE_CLASS(Codegen, (),
                       (lanes, opt_level, codegen_level, slp_vectorize,
                        loop_vectorize, veclib, contract),
                       (), ())
};

// The identity, and around it the policy: whether to compile, and for what
// batch.  None of the policy reaches the emitter.
struct Options {
  // Read by Equation, never by Compiler::compile().
  Backend backend = Backend::Interpret;
  // The batch one call will carry.  Stated, not inferred: the kernel is built
  // before any call exists to read an `n` from.  Sets the lane width only.
  std::size_t points = 1;
  Codegen codegen{};
  bool time_passes = false; // Per-pass timing to stderr

  friend bool operator==(const Options &, const Options &) = default;
};

// The options a lane compiles under: a batch too short to fill one block of
// the sweep emits scalar, so kernel and sweep agree on where a batch stops
// being a batch.  A stated width is honoured.
[[nodiscard]] constexpr Options for_batch(Options opt,
                                          std::size_t block_lanes) noexcept {
  if (!opt.codegen.lanes.stated() && opt.points < block_lanes) {
    opt.codegen.lanes = Lanes::scalar();
  }
  return opt;
}

// The column counts a kernel is called with, read off the graph it was
// compiled from.  One value, so the four cannot be handed over transposed.
// Flat, rather than holding the `rt::Layout` it is copied from: jit is core,
// and the core may not include rt/ -- which is why `rt::Graph` is only
// forward-declared above and `of` is a template.  The three names repeat here
// because the layering says they must, not because nobody noticed.
struct KernelShape {
  std::size_t arity = 0;
  std::size_t values = 0;
  std::size_t jacobian = 0;
  std::size_t hessian = 0;
  // Elements of T in the interpreter's two tapes, `lanes` wide, which a
  // kernel's tensor steps sweep through.  Zero for a scalar graph.
  std::size_t tapes = 0;

  template <impl::Numeric T>
  [[nodiscard]] static KernelShape of(const rt::Graph<T> &g,
                                      std::size_t lanes = 1) {
    const auto &layout = g.layout();
    return KernelShape{
        .arity = g.arity(),
        .values = layout.values,
        .jacobian = layout.jacobian,
        .hessian = layout.hessian,
        .tapes =
            g.has_tensors() ? (g.tape_rows() + g.tensors().extent) * lanes : 0};
  }
  [[nodiscard]] constexpr std::size_t outputs() const noexcept {
    return values + jacobian + hessian;
  }
  friend constexpr bool operator==(KernelShape, KernelShape) noexcept = default;
};

} // namespace ddx::jit
