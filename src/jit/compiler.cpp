#include "codegen.hpp"

#include "spelling.hpp"

#include "rt/interpret.hpp"
#include "rt/pipeline.hpp"

#include <boost/hash2/fnv1a.hpp>
#include <boost/hash2/hash_append.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/scope/scope_exit.hpp>

#include <llvm/Config/llvm-config.h>

#include <llvm/ADT/StringMap.h>

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/ReplaceWithVeclib.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassTimingInfo.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstring>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ddx::jit {
namespace {

// Global LLVM state, and a process may hold more than one Compiler.
void init_native_target_once() {
  [[maybe_unused]] static const bool once = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
}

// The one seam between LLVM's error model and ours.
[[nodiscard]] compile_error as_error(const errc code, const llvm::Twine &what,
                                     llvm::Error e) {
  return compile_error{.code = code,
                       .detail =
                           (what + ": " + llvm::toString(std::move(e))).str()};
}

// A Call row of the opcode tables as the runtime symbol its spelling names at
// T, through the functor the row carries; a row spelled any other way defines
// nothing.  A template, so the other arity's lambda is never formed.
template <CKernelScalar T, rt::OpCode Op, typename Functor>
void define_call(auto &def) {
  if constexpr (rt::spelling_of(Op).form == rt::Form::Call) {
    if constexpr (rt::arity_of(Op) == 1) {
      def(
          detail::libm_name<T>(rt::spelling_of(Op).name),
          +[](T u) noexcept { return T{Functor{}(u)}; });
    } else {
      def(
          detail::libm_name<T>(rt::spelling_of(Op).name),
          +[](T l, T r) noexcept { return T{Functor{}(l, r)}; });
    }
  }
}

// A kernel over tensor rows is emitted at a power of two up to this, the
// widths the interpreter's tensor step is instantiated at.
constexpr std::size_t tensor_widths = 5;
constexpr unsigned widest_tensor_lanes = 1u << (tensor_widths - 1);

// The interpreter's step for node v, over the kernel's tapes: the TapeView
// Core::interpret builds, at the kernel's width.
template <CKernelScalar T>
void tensor_step(const rt::detail::TensorCall<T> &c, rt::NodeId v,
                 std::size_t lanes, T *tape, T *vtape) {
  const rt::Graph<T> &g = *c.graph;
  const rt::detail::TapeView<T> t{.scalar_rows = g.rows(),
                                  .tensor_rows = g.tensors().rows,
                                  .vtape = vtape,
                                  .kernels = c.kernels};
  boost::mp11::mp_with_index<tensor_widths>(
      static_cast<std::size_t>(std::countr_zero(lanes)), [&](auto i) {
        constexpr std::size_t W = std::size_t{1} << decltype(i)::value;
        const auto lane =
            rt::detail::lanes_of<W>(std::span<T>{tape, g.tape_rows() * W});
        rt::detail::lanes_tensor<W>(*c.arena, v, (*c.arena)[v],
                                    lane(g.rows()[v]), lane, t);
      });
}

// The sweep's scatter of every output block, `live` of the kernel's lanes.
template <CKernelScalar T>
void tensor_store(const rt::detail::TensorCall<T> &c, T *const *f, T *const *g,
                  T *const *h, std::size_t i, std::size_t live,
                  std::size_t lanes, const T *tape, const T *vtape) {
  const rt::Graph<T> &graph = *c.graph;
  const rt::Layout &layout = graph.layout();
  rt::scatter_blocks(graph.output_rows(),
                     std::span<const T>{tape, graph.tape_rows() * lanes},
                     std::span<const T>{vtape, graph.tensors().extent * lanes},
                     rt::Columns<T>{.values = {f, layout.values},
                                    .jacobian = {g, layout.jacobian},
                                    .hessian = {h, layout.hessian}},
                     i, lanes, live);
}

// Every name a T kernel calls out to: sin for double, sinf for float.
template <CKernelScalar T> void define_scalar(auto &def) {
#define DDX_JIT_RUNTIME(fn, Op, label, functor, ...)                           \
  define_call<T, rt::OpCode::Op, functor>(def);
  DDX_RT_UNARY_TABLE(DDX_JIT_RUNTIME)
  DDX_RT_BINARY_TABLE(DDX_JIT_RUNTIME)
#undef DDX_JIT_RUNTIME
#define DDX_JIT_RUNTIME(fn, Op, label)                                         \
  def(                                                                         \
      detail::libm_name<T>(rt::spelling_of(rt::OpCode::Op).name),              \
      +[](T u) noexcept { return std::fn(u); });
  DDX_UNARY_MATH_TABLE(DDX_JIT_RUNTIME)
#undef DDX_JIT_RUNTIME
  def(
      detail::libm_name<T>(detail::trampoline_symbol),
      +[](const void *fn, const T *in, std::uint64_t arity, T *answer,
          std::uint64_t components) noexcept {
        (*static_cast<const rt::ExternFn<T> *>(fn))(
            {in, static_cast<std::size_t>(arity)},
            {answer, static_cast<std::size_t>(components)});
      });
  def(
      detail::libm_name<T>(detail::tensor_step_symbol),
      +[](const rt::detail::TensorCall<T> *c, std::uint64_t node,
          std::uint64_t lanes, T *tape, T *vtape) noexcept {
        tensor_step(*c, static_cast<rt::NodeId>(node),
                    static_cast<std::size_t>(lanes), tape, vtape);
      });
  def(
      detail::libm_name<T>(detail::tensor_store_symbol),
      +[](const rt::detail::TensorCall<T> *c, T *const *f, T *const *g,
          T *const *h, std::uint64_t i, std::uint64_t live, std::uint64_t lanes,
          const T *tape, const T *vtape) noexcept {
        tensor_store(*c, f, g, h, static_cast<std::size_t>(i),
                     static_cast<std::size_t>(live),
                     static_cast<std::size_t>(lanes), tape, vtape);
      });
}

// What a kernel calls that is not an intrinsic: every Call row of the
// spelling table, through the functor its opcode row names, and the extern
// trampoline.  Defined outright: GetForCurrentProcess resolves whatever a
// loaded module exports, which need not be the libm this TU was compiled
// against.
[[nodiscard]] llvm::Error define_runtime(llvm::orc::ExecutionSession &es,
                                         llvm::orc::JITDylib &jd,
                                         const llvm::DataLayout &dl) {
  llvm::orc::MangleAndInterner mangle(es, dl);
  llvm::orc::SymbolMap syms;
  syms.reserve(2 * (rt::op_count + 3));
  // The unary + turns each lambda into a plain function pointer.
  const auto def = [&](const std::string &name, auto *fn) {
    syms[mangle(name)] = {llvm::orc::ExecutorAddr::fromPtr(fn),
                          llvm::JITSymbolFlags::Exported |
                              llvm::JITSymbolFlags::Callable};
  };
  define_scalar<double>(def);
  define_scalar<float>(def);
  return jd.define(llvm::orc::absoluteSymbols(std::move(syms)));
}

// In *this process*, not whether the target could have one: promising a vector
// sin that cannot be called fails at link.
[[nodiscard]] bool want_veclib(const llvm::Triple &triple, const Options &opt,
                               const bool have) {
  return have && (opt.codegen.veclib == VecLib::Libmvec ||
                  (opt.codegen.veclib == VecLib::Auto && triple.isOSLinux() &&
                   triple.getArch() == llvm::Triple::x86_64));
}

llvm::TargetLibraryInfoImpl target_library_info(const llvm::Triple &triple,
                                                const Options &opt,
                                                const bool have) {
  llvm::TargetLibraryInfoImpl tlii(triple);
  if (want_veclib(triple, opt, have)) {
    tlii.addVectorizableFunctionsFromVecLib(
        llvm::TargetLibraryInfoImpl::LIBMVEC_X86, triple);
  }
  return tlii;
}

// The widest fixed width the vector library serves a transcendental at T: LLVM
// 20's libmvec table stops at four doubles, so an AVX-512 host's eight-wide
// intrinsic has no library form and unrolls to scalar calls.
template <CKernelScalar T>
[[nodiscard]] unsigned library_lanes(const llvm::Triple &triple) {
  const auto tlii = target_library_info(
      triple, Options{.codegen = {.veclib = VecLib::Libmvec}}, true);
  llvm::ElementCount fixed;
  llvm::ElementCount scalable;
  tlii.getWidestVF(detail::libm_name<T>("sin"), fixed, scalable);
  return std::max(1u, static_cast<unsigned>(fixed.getFixedValue()));
}

// A derived width is the host's, or under a vector library the widest it
// serves, so it never asks for a call that is not there.  A stated width is
// taken as stated.
[[nodiscard]] unsigned emitted_lanes(const Options &opt,
                                     const std::string &triple,
                                     const bool have_libmvec,
                                     const unsigned host, const unsigned lib) {
  return opt.codegen.lanes.stated().value_or(
      want_veclib(llvm::Triple{triple}, opt, have_libmvec) ? std::min(host, lib)
                                                           : host);
}

// On the module rather than the JIT, so one JIT serves compiles at different
// levels.
constexpr const char *kCodegenFlag = "ddx.codegen";

[[nodiscard]] std::optional<std::uint64_t> flag_of(const llvm::Module &m,
                                                   const char *key) {
  const auto *const flag =
      llvm::mdconst::extract_or_null<llvm::ConstantInt>(m.getModuleFlag(key));
  return flag != nullptr ? std::optional{flag->getZExtValue()} : std::nullopt;
}

// The flag is written from std::to_underlying(OptLevel), which is LLVM's own
// 0..3 scale.
llvm::CodeGenOptLevel codegen_level_of(const llvm::Module &m) {
  return llvm::CodeGenOpt::getLevel(
             static_cast<int>(flag_of(m, kCodegenFlag).value_or(2)))
      .value_or(llvm::CodeGenOptLevel::Default);
}

// One TargetMachine per module, which is what makes two compiles at once safe:
// the default SimpleCompiler shares one whose subtarget cache is
// unsynchronised.
class LevelledCompiler final : public llvm::orc::IRCompileLayer::IRCompiler {
public:
  explicit LevelledCompiler(llvm::orc::JITTargetMachineBuilder machine)
      : IRCompiler(llvm::orc::irManglingOptionsFromTargetOptions(
            machine.getOptions())),
        machine_(std::move(machine)) {}

  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  operator()(llvm::Module &m) override {
    auto machine = machine_;
    machine.setCodeGenOptLevel(codegen_level_of(m));
    auto tm = machine.createTargetMachine();
    return (!tm) ? tm.takeError() : llvm::orc::SimpleCompiler{**tm}(m);
  }

private:
  llvm::orc::JITTargetMachineBuilder machine_;
};

// The host as one line.  The feature string is folded, not spelled:
// detectHost() answers with 1.5 KB of `+avx2,-avx512f,...`.
[[nodiscard]] std::string
host_identity_of(const llvm::orc::JITTargetMachineBuilder &machine) {
  const std::string features = machine.getFeatures().getString();
  boost::hash2::fnv1a_64 h;
  h.update(features.data(), features.size());
  const llvm::StringRef cpu = machine.getCPU();
  return std::format("{}/{}/feat-{:016x}/llvm-" LLVM_VERSION_STRING "/abi-2",
                     machine.getTargetTriple().str(),
                     cpu.empty() ? std::string_view{"generic"}
                                 : std::string_view{cpu.data(), cpu.size()},
                     h.result());
}

// Asked of the target's cost model rather than a feature list, so
// prefer-256-bit parts answer 256 as the backend would split them anyway.
[[nodiscard]] unsigned
vector_bits(llvm::orc::JITTargetMachineBuilder machine) {
  auto tm = machine.createTargetMachine();
  if (!tm) {
    llvm::consumeError(tm.takeError());
    return 64;
  }
  llvm::LLVMContext ctx;
  llvm::Module probe("ddx.probe", ctx);
  auto *const fn = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false),
      llvm::Function::ExternalLinkage, "probe", probe);
  const llvm::TypeSize bits =
      (*tm)->getTargetTransformInfo(*fn).getRegisterBitWidth(
          llvm::TargetTransformInfo::RGK_FixedWidthVector);
  return std::max(64u, static_cast<unsigned>(bits.getFixedValue()));
}

// One straight-line block the graph has already shared, so all the middle end
// can do is fold.  A vector library rewrites the lane intrinsics to its own
// entry points here; with none the backend unrolls them into scalar libm calls.
[[nodiscard]] llvm::Error optimize(llvm::Module &m, llvm::TargetMachine &tm,
                                   const Options &opt,
                                   const bool have_libmvec) {
  const llvm::Triple triple(m.getTargetTriple());
  llvm::TargetLibraryInfoImpl tlii =
      target_library_info(triple, opt, have_libmvec);

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  // A disabled handler registers nothing, so this is the same pipeline.
  llvm::PassInstrumentationCallbacks pic;
  llvm::TimePassesHandler timers(opt.time_passes);
  timers.setOutStream(llvm::errs());
  timers.registerCallbacks(pic);

  // The loop is already emitted `lanes` wide, and the loop vectoriser pays for
  // an alias check between every pair of columns and then declines.
  llvm::PipelineTuningOptions pto;
  pto.SLPVectorization = opt.codegen.slp_vectorize;
  pto.LoopVectorization = opt.codegen.loop_vectorize;

  llvm::PassBuilder pb(&tm, pto, std::nullopt, &pic);
  // Before registerFunctionAnalyses, which would otherwise install the default
  // llvm::TargetLibraryAnalysis and with it an empty vector-function table.
  fam.registerPass([&] { return llvm::TargetLibraryAnalysis(tlii); });
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  // Not constexpr in LLVM, so a static table rather than a switch.
  static const std::array levels{
      llvm::OptimizationLevel::O0, llvm::OptimizationLevel::O1,
      llvm::OptimizationLevel::O2, llvm::OptimizationLevel::O3};
  const llvm::OptimizationLevel level =
      levels[std::to_underlying(opt.codegen.opt_level)];

  llvm::ModulePassManager mpm = level == llvm::OptimizationLevel::O0
                                    ? pb.buildO0DefaultPipeline(level)
                                    : pb.buildPerModuleDefaultPipeline(level);
  if (want_veclib(triple, opt, have_libmvec)) {
    llvm::FunctionPassManager fpm;
    fpm.addPass(llvm::ReplaceWithVeclib());
    mpm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
  }
  mpm.run(m, mam);
  return llvm::Error::success();
}

using Clock = std::chrono::steady_clock;

// `code` is what a Kernel holds a share of, so it outlives every Compiler that
// named it.  The machine is a *recipe*: TargetMachine caches subtargets in an
// unsynchronised map, so two compiles must not share one.
struct Host {
  llvm::orc::LLJIT &jit;
  llvm::orc::JITTargetMachineBuilder machine;
  std::string triple;
  bool libmvec;
  unsigned lanes;        // What a derived Options::lanes means here
  unsigned veclib_lanes; // and under a vector library
  std::shared_ptr<void> code;

  [[nodiscard]] unsigned lanes_for(const Options &opt) const {
    return emitted_lanes(opt, triple, libmvec, lanes, veclib_lanes);
  }
};

// One graph compiled once, filling the report it was handed as it goes.
template <CKernelScalar T> class CompileJob : private impl::pinned {
public:
  CompileJob(Host host, const rt::Graph<T> &g, const Options &opt,
             std::string name, CompileReport &into)
      : host_(std::move(host)), g_(g), opt_(opt), name_(std::move(name)),
        rep_(into) {}

  [[nodiscard]] compile_result<llvm::orc::ThreadSafeModule> prepared() {
    return emitted().and_then([this](llvm::orc::ThreadSafeModule m) {
      return optimized(std::move(m));
    });
  }

  [[nodiscard]] compile_result<Kernel<T>> operator()() {
    return prepared().and_then([this](llvm::orc::ThreadSafeModule m) {
      return materialised(std::move(m));
    });
  }

private:
  [[nodiscard]] compile_result<llvm::orc::ThreadSafeModule> emitted() {
    const boost::scope::scope_exit clock{
        [this, start = Clock::now()] { rep_.emit = Clock::now() - start; }};
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto m = detail::emit_module(*ctx, g_, name_, lanes(),
                                 host_.jit.getDataLayout(), host_.triple);
    if (!m) {
      // The emitter has already written the verifier's own diagnosis.
      return std::unexpected{compile_error{
          errc::jit_verify, "the emitted module failed verification"}};
    }
    // LLVM prints this in diagnostics; the kernel's name is unique per compile.
    m->setModuleIdentifier(name_);
    m->addModuleFlag(llvm::Module::Error, kCodegenFlag,
                     std::to_underlying(opt_.codegen.codegen_level));
    rep_.nodes = g_.live_count();
    rep_.instructions = m->getInstructionCount();
    return llvm::orc::ThreadSafeModule{
        std::move(m), llvm::orc::ThreadSafeContext{std::move(ctx)}};
  }

  [[nodiscard]] compile_result<llvm::orc::ThreadSafeModule>
  optimized(llvm::orc::ThreadSafeModule m) {
    const boost::scope::scope_exit clock{
        [&, start = Clock::now()] { rep_.optimize = Clock::now() - start; }};
    // One machine per compile -- see Host.
    auto tm = host_.machine.createTargetMachine();
    if (!tm) {
      return std::unexpected{as_error(
          errc::jit_target, "creating a target machine", tm.takeError())};
    }
    std::optional<compile_error> refused;
    m.withModuleDo([&](llvm::Module &mod) {
      if (auto e = optimize(mod, **tm, opt_, host_.libmvec)) {
        refused = as_error(errc::jit_module, "building the pass pipeline",
                           std::move(e));
      }
    });

    if (refused) {
      return std::unexpected{*refused};
    }
    return m;
  }

  // The backend runs on materialisation, so the lookup is inside this phase or
  // its number is wrong.
  [[nodiscard]] compile_result<Kernel<T>>
  materialised(llvm::orc::ThreadSafeModule m) {
    const boost::scope::scope_exit clock{
        [this, start = Clock::now()] { rep_.codegen = Clock::now() - start; }};
    if (auto e = host_.jit.addIRModule(std::move(m))) {
      return std::unexpected{
          as_error(errc::jit_module, "adding the module", std::move(e))};
    }
    auto sym = host_.jit.lookup(name_);
    if (!sym) {
      return std::unexpected{as_error(errc::jit_lookup,
                                      llvm::Twine{"looking up "} + name_,
                                      sym.takeError())};
    }
    return Kernel<T>{sym->toPtr<typename Kernel<T>::function_type>(),
                     KernelShape::of(g_, lanes()), Externs::of(g_),
                     std::move(host_.code)};
  }

  // Every width gives the same bits, so a tensor graph rounds down to one its
  // step is instantiated at.
  [[nodiscard]] unsigned lanes() const {
    const unsigned w = host_.lanes_for(opt_);
    return g_.has_tensors() ? std::bit_floor(std::min(w, widest_tensor_lanes))
                            : w;
  }

  Host host_;
  const rt::Graph<T> &g_;
  const Options &opt_;
  std::string name_;
  CompileReport &rep_;
};

} // namespace

struct Compiler::Impl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  // Copied before the builder goes to LLJIT: every compile stamps its own.
  std::optional<llvm::orc::JITTargetMachineBuilder> machine;
  std::string triple;
  // Folded once at bring-up: the builder it comes from goes to LLJIT and no
  // longer answers for the host.
  std::string host;
  // The only shared mutable state: LLJIT is synchronised, but two threads
  // naming a module alike would hand it a duplicate symbol.
  std::atomic<unsigned> counter{0};
  bool libmvec = false; // Whether the vector forms resolve here
  unsigned bits = 64;   // The host's vector register, in bits
  // The widest the vector library serves: double's, then float's.
  std::array<unsigned, 2> veclib_lanes{1, 1};

  [[nodiscard]] static compile_result<std::shared_ptr<Impl>> bring_up() {
    init_native_target_once();
    auto impl = std::make_shared<Impl>();

    // Parity with the project's -march=native.
    auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmb) {
      return std::unexpected{
          as_error(errc::jit_target, "detecting the host", jtmb.takeError())};
    }
    impl->machine = *jtmb;
    impl->triple = jtmb->getTargetTriple().str();
    impl->bits = vector_bits(*jtmb);
    impl->veclib_lanes = {library_lanes<double>(jtmb->getTargetTriple()),
                          library_lanes<float>(jtmb->getTargetTriple())};
    impl->host = host_identity_of(*jtmb);

    // Not setNumCompileThreads: the backend runs on the thread that asked, and
    // no idle pool is spawned.
    auto compiler_factory =
        [](llvm::orc::JITTargetMachineBuilder machine)
        -> llvm::Expected<
            std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
      return std::make_unique<LevelledCompiler>(std::move(machine));
    };
    if (auto jit = llvm::orc::LLJITBuilder()
                       .setJITTargetMachineBuilder(std::move(*jtmb))
                       .setCompileFunctionCreator(std::move(compiler_factory))
                       .create();
        !jit) {
      return std::unexpected{
          as_error(errc::jit_target, "creating the JIT", jit.takeError())};
    } else {
      impl->jit = std::move(*jit);
    }

    llvm::orc::JITDylib &jd = impl->jit->getMainJITDylib();
    const llvm::DataLayout &dl = impl->jit->getDataLayout();
    const char prefix = dl.getGlobalPrefix();

    // Before the generators: a definition here beats anything they would find.
    if (auto e = define_runtime(impl->jit->getExecutionSession(), jd, dl)) {
      return std::unexpected{
          as_error(errc::jit_target, "defining libm", std::move(e))};
    }

    if (auto procs =
            llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                prefix);
        !procs) {
      return std::unexpected{as_error(
          errc::jit_target, "opening the process symbols", procs.takeError())};
    } else {
      jd.addGenerator(std::move(*procs));
    }

    // Not loaded in a program that never called it, so the generator above
    // cannot see it -- and loaded even where the default declines to use it.
    if (auto vec = llvm::orc::DynamicLibrarySearchGenerator::Load(
            "libmvec.so.1", prefix)) {
      jd.addGenerator(std::move(*vec));
      impl->libmvec = true;
    } else {
      // Not an error: it leaves the transcendentals scalar, nothing more.
      llvm::consumeError(vec.takeError());
    }
    return impl;
  }

  // On Impl, so a queued compile needs only a share of this and not a Compiler
  // that may have been moved from.
  // What a CompileJob is handed: the process-wide JIT, and a share of this,
  // so the code outlives any Compiler that goes away mid-compile.
  template <CKernelScalar T>
  [[nodiscard]] static Host host_of(const std::shared_ptr<Impl> &self) {
    return Host{.jit = *self->jit,
                .machine = *self->machine,
                .triple = self->triple,
                .libmvec = self->libmvec,
                .lanes = self->bits / unsigned{8 * sizeof(T)},
                .veclib_lanes = self->veclib_lanes[std::same_as<T, float>],
                .code = self};
  }

  template <CKernelScalar T>
  [[nodiscard]] static compile_result<Kernel<T>>
  run(const std::shared_ptr<Impl> &self, const rt::Graph<T> &g,
      const Options &opt, CompileReport &rep) {
    return CompileJob<T>{host_of<T>(self), g, opt,
                         "ddx_kernel_" + std::to_string(self->counter++),
                         rep}();
  }
};

Compiler::Compiler(std::shared_ptr<Impl> impl) noexcept
    : impl_{std::move(impl)} {}

std::string_view Compiler::host_identity() const noexcept {
  return impl_->host;
}

compile_result<Compiler> Compiler::create() {
  return Impl::bring_up().transform(
      [](std::shared_ptr<Impl> impl) { return Compiler{std::move(impl)}; });
}
Compiler::~Compiler() = default;
Compiler::Compiler(Compiler &&) noexcept = default;
Compiler &Compiler::operator=(Compiler &&) noexcept = default;

template <CKernelScalar T>
compile_result<Kernel<T>> Compiler::compile(const rt::Graph<T> &g,
                                            const Options &opt,
                                            CompileReport *report) {
  CompileReport discard;
  return Impl::run(impl_, g, opt, report != nullptr ? *report : discard);
}

template <CKernelScalar T>
compile_result<Kernel<T>> Compiler::adopt(std::span<const std::byte>,
                                          std::string_view,
                                          const rt::Graph<T> &) {
  return std::unexpected{
      compile_error{errc::jit_object, "objects are not kept"}};
}

template <CKernelScalar T>
compile_result<std::string> Compiler::render_ir(const rt::Graph<T> &g,
                                                const Options &opt) const {
  CompileReport discard;
  CompileJob<T> run{Impl::host_of<T>(impl_), g, opt, "ddx_kernel_dump",
                    discard};
  return run.prepared().transform([](llvm::orc::ThreadSafeModule m) {
    std::string out;
    llvm::raw_string_ostream os(out);
    m.withModuleDo([&](llvm::Module &mod) { mod.print(os, nullptr); });
    return out;
  });
}

#define DDX_JIT_INSTANCES(T)                                                   \
  template DDX_INSTANCE_API compile_result<Kernel<T>> Compiler::compile(       \
      const rt::Graph<T> &, const Options &, CompileReport *);                 \
  template DDX_INSTANCE_API compile_result<Kernel<T>> Compiler::adopt(         \
      std::span<const std::byte>, std::string_view, const rt::Graph<T> &);     \
  template DDX_INSTANCE_API compile_result<std::string> Compiler::render_ir(   \
      const rt::Graph<T> &, const Options &) const;
DDX_JIT_INSTANCES(double)
DDX_JIT_INSTANCES(float)
#undef DDX_JIT_INSTANCES

} // namespace ddx::jit
