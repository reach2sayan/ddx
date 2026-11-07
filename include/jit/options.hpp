#pragma once

#include "ops/numeric.hpp"
#include "util/error.hpp"

#include <boost/describe.hpp>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>

// Compile options only, present in every build so an Equation has one shape
// with or without the JIT. jit/kernel.hpp holds the LLVM side.
namespace ddx::rt {
template <impl::Numeric T> class Graph;
namespace detail {
template <impl::Numeric T> struct TensorCall;
} // namespace detail
} // namespace ddx::rt

namespace ddx::jit {

// Off by default, since glibc's vector routines are ~4 ULP against ~0.5 scalar.
enum class VecLib : std::uint8_t { None, Libmvec };
BOOST_DESCRIBE_ENUM(VecLib, None, Libmvec)

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

// LLVM's -O0 to -O3, used for both the IR pipeline and codegen.
enum class OptLevel : std::uint8_t { O0, O1, O2, O3 };
BOOST_DESCRIBE_ENUM(OptLevel, O0, O1, O2, O3)

static_assert(DDX_JIT_DEFAULT_OPT >= 0 && DDX_JIT_DEFAULT_OPT <= 3);
inline constexpr OptLevel default_opt_level =
    static_cast<OptLevel>(DDX_JIT_DEFAULT_OPT);
inline constexpr bool default_contract = DDX_JIT_DEFAULT_CONTRACT != 0;

// Points per loop iteration; derived means the host's register width. Every
// width gives the same bits.
class Lanes {
public:
  constexpr Lanes() = default;
  [[nodiscard]] static constexpr Lanes derived() noexcept { return {}; }
  [[nodiscard]] static constexpr Lanes scalar() noexcept { return Lanes{1}; }
  // A stated width must be at least 1; zero gives nullopt.
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
};

// Compile builds each kernel on the first call that needs it, which waits.
enum class Backend : std::uint8_t { Interpret, Compile };
BOOST_DESCRIBE_ENUM(Backend, Interpret, Compile)

// Everything that decides the machine code, and nothing else.
struct Codegen {
  Lanes lanes = Lanes::derived();
  OptLevel opt_level = default_opt_level;
  // Kept low, since selection and register allocation are ~95% of a compile.
  OptLevel codegen_level = OptLevel::O1;
  // Vectorises within one point; off because the gain depends on the model.
  bool slp_vectorize = false;
  // With a vector library, derived lanes become the widest width it serves,
  // four doubles for libmvec.
  VecLib veclib = VecLib::None;
  bool contract = default_contract; // follows DDX_FP_FLAGS

  friend bool operator==(const Codegen &, const Codegen &) = default;
};

// Codegen plus the policy of whether to compile and for what batch. The policy
// never reaches the emitter.
struct Options {
  // Read by Equation, never by Compiler::compile().
  Backend backend = Backend::Interpret;
  // The batch size one call will carry, stated up front because the kernel is
  // built before any call. It only sets the lane width.
  std::size_t points = 1;
  Codegen codegen{};

  friend bool operator==(const Options &, const Options &) = default;
};

// A batch too short to fill one sweep block compiles scalar, so kernel and
// sweep agree. A stated lane width is kept.
[[nodiscard]] constexpr Options for_batch(Options opt,
                                          std::size_t block_lanes) noexcept {
  if (!opt.codegen.lanes.stated() && opt.points < block_lanes) {
    opt.codegen.lanes = Lanes::scalar();
  }
  return opt;
}

// The column counts a kernel is called with, read off its graph. A flat copy of
// rt::Layout, because jit may not include rt/.
struct KernelShape {
  std::size_t arity = 0;
  std::size_t values = 0;
  std::size_t jacobian = 0;
  std::size_t hessian = 0;
  // Elements of T in the interpreter's two tapes, lanes wide, for the tensor
  // steps. Zero for a scalar graph.
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
