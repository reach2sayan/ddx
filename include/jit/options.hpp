#pragma once

#include "ops/numeric.hpp"
#include "util/error.hpp"

#include <boost/describe.hpp>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>

// What a caller says about compiling, and nothing that compiles: header types
// in every build, so an Equation has one shape whether or not a backend was
// compiled in.  jit/kernel.hpp holds the LLVM side, cl/ the device side.
namespace ddx::rt {
template <impl::Numeric T> class Graph;
} // namespace ddx::rt

namespace ddx::jit {

// Off: glibc's vector routines are ~4 ULP where the scalar ones are ~0.5.
enum class VecLib : std::uint8_t { None, Auto, Libmvec };

struct error {
  errc code;
  std::string detail;
};

template <typename T> using result = std::expected<T, error>;

#ifndef DDX_JIT_DEFAULT_OPT
#define DDX_JIT_DEFAULT_OPT 2
#endif
#ifndef DDX_JIT_DEFAULT_CONTRACT
#define DDX_JIT_DEFAULT_CONTRACT 1
#endif

// LLVM's -O0 to -O3, for the IR pipeline and for codegen.
enum class Level : std::uint8_t { O0, O1, O2, O3 };

static_assert(DDX_JIT_DEFAULT_OPT >= 0 && DDX_JIT_DEFAULT_OPT <= 3);
inline constexpr Level default_opt_level =
    static_cast<Level>(DDX_JIT_DEFAULT_OPT);
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

// Compile and Adapt are a ladder: a codegen-0 kernel answers first and one at
// codegen_level replaces it, bit-identically; calls before the first rung lands
// are swept.  Adapt asks for a rung once the batch has paid for it.  Device
// builds the graph as OpenCL C for the device Options::device names -- one
// rung, and the sweep answers until it lands.  Appended, so a saved byte keeps
// its meaning.
enum class Backend : std::uint8_t { Interpret, Compile, Adapt, Device };

// What decides the machine code, and nothing else: the identity a stored
// object is matched against and the object cache is keyed on, so a field here
// is a field of both by construction.
struct Codegen {
  Lanes lanes = Lanes::derived();
  Level opt_level = default_opt_level;
  // Under Backend::Compile the top rung, so O0 asks for a single one.
  // Selection and register allocation are ~95% of a compile.
  Level codegen_level = Level::O1;
  // Within one point; model-dependent, hence off.
  bool slp = false;
  // Off: the body is already `lanes` wide, and past ~19 columns the alias
  // checks exceed their own budget.
  bool loop_vectorize = false;
  // With one, a derived `lanes` is the widest width the library serves -- four
  // doubles for libmvec -- whatever the host's registers hold.
  VecLib veclib = VecLib::None;
  bool contract = default_contract; // Follows DDX_FP_FLAGS

  friend bool operator==(const Codegen &, const Codegen &) = default;
  BOOST_DESCRIBE_CLASS(Codegen, (),
                       (lanes, opt_level, codegen_level, slp, loop_vectorize,
                        veclib, contract),
                       (), ())
};

// The identity, and around it the policy: whether and when to compile, what to
// keep.  None of the policy reaches the emitter.
struct Options {
  // Read by Equation, never by Compiler::compile().
  Backend backend = Backend::Interpret;
  // The batch one call will carry.  Stated, not inferred: the kernel is built
  // before any call exists to read an `n` from.  Sets the lane width only.
  std::size_t points = 1;
  Codegen codegen{};
  // Points, not points x nodes: a compile costs about what a node costs and a
  // swept point saves about what a node saves, so the size cancels out.
  std::size_t warm_points = 1uz << 16;
  std::size_t hot_points = 1uz << 20;
  bool time_passes = false; // Per-pass timing to stderr
  // On: the object is the one part of a saved equation nothing can reconstruct.
  bool retain_object = true;
  // Object cache directory, empty for none; keyed by graph, options and host.
  // Not serialised: a path is this machine's.
  std::string cache_dir{};
  // Under Backend::Device, which OpenCL device: a case-insensitive substring of
  // "<platform> / <device>".  Empty is the first GPU with double precision,
  // else the first double-precision device of any type.  Not serialised, for
  // the reason cache_dir is not: which devices exist is this machine's.
  std::string device{};

  friend bool operator==(const Options &, const Options &) = default;
  BOOST_DESCRIBE_CLASS(Options, (),
                       (backend, points, codegen, warm_points, hot_points,
                        time_passes, retain_object),
                       (), ())
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

  template <impl::Numeric T>
  [[nodiscard]] static KernelShape of(const rt::Graph<T> &g) {
    const auto &layout = g.layout();
    return KernelShape{.arity = g.arity(),
                       .values = layout.values,
                       .jacobian = layout.jacobian,
                       .hessian = layout.hessian};
  }
  [[nodiscard]] constexpr std::size_t outputs() const noexcept {
    return values + jacobian + hessian;
  }
  friend constexpr bool operator==(KernelShape, KernelShape) noexcept = default;
};

} // namespace ddx::jit
