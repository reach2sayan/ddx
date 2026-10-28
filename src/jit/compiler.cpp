#include "codegen.hpp"

#include "spelling.hpp"

#include "rt/interpret.hpp"
#include "rt/pipeline.hpp"

#include <boost/mp11/algorithm.hpp>

#include <llvm/ADT/StringMap.h>

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/ReplaceWithVeclib.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
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
#include <concepts>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>

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
  const auto t = rt::detail::TapeView<T>::of(g, vtape, c.kernels);
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
  def(
      detail::libm_name<T>(detail::tensor_table_symbol),
      +[](const rt::detail::TensorCall<T> *c, std::uint64_t ref) noexcept {
        return c->arena->tensors()[static_cast<std::size_t>(ref)].data.data();
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
void optimize(llvm::Module &m, llvm::TargetMachine &tm, const Options &opt,
              const bool have_libmvec) {
  const llvm::Triple &triple = tm.getTargetTriple();
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
}

// A tensor loop is marked for the vectoriser, which says so on stderr when it
// declines one; that is the loop running scalar, not an error.
void quiet_vectoriser(llvm::LLVMContext &ctx) {
  ctx.setDiagnosticHandlerCallBack([](const llvm::DiagnosticInfo *d, void *) {
    if (d->getKind() == llvm::DK_OptimizationFailure) {
      return;
    }
    llvm::DiagnosticPrinterRawOStream out(llvm::errs());
    llvm::errs() << llvm::LLVMContext::getDiagnosticMessagePrefix(
                        d->getSeverity())
                 << ": ";
    d->print(out);
    llvm::errs() << '\n';
  });
}

} // namespace

// `self` is what a Kernel holds a share of, so the code outlives every
// Compiler that named it.
struct Compiler::Impl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  // A recipe, not a machine: TargetMachine caches subtargets in an
  // unsynchronised map, so every compile stamps its own.
  std::optional<llvm::orc::JITTargetMachineBuilder> machine;
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
    impl->bits = vector_bits(*jtmb);
    impl->veclib_lanes = {library_lanes<double>(jtmb->getTargetTriple()),
                          library_lanes<float>(jtmb->getTargetTriple())};

    // Not setNumCompileThreads: the backend runs on the thread that asked, and
    // no idle pool is spawned.  The recipe is copied once the builder has set
    // what its linker needs -- JITLink's small code model and PIC; the JIT
    // default is the large model, a 64-bit load per constant.
    llvm::orc::LLJITBuilder builder;
    builder.setJITTargetMachineBuilder(std::move(*jtmb));
    if (auto e = builder.prepareForConstruction()) {
      return std::unexpected{
          as_error(errc::jit_target, "preparing the JIT", std::move(e))};
    }
    impl->machine = *builder.JTMB;
    if (auto jit = builder.create(); !jit) {
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

  // A derived width is the host's, or under a vector library the widest it
  // serves, so it never asks for a call that is not there; a stated width is
  // taken as stated.  Every width gives the same bits, so a tensor graph
  // rounds down to one its step is instantiated at.
  template <CKernelScalar T>
  [[nodiscard]] unsigned lanes(const Options &opt, const bool tensors) const {
    const unsigned host = bits / unsigned{8 * sizeof(T)};
    const unsigned w = opt.codegen.lanes.stated().value_or(
        want_veclib(machine->getTargetTriple(), opt, libmvec)
            ? std::min(host, veclib_lanes[std::same_as<T, float>])
            : host);
    return tensors ? std::bit_floor(std::min(w, widest_tensor_lanes)) : w;
  }

  // Emitted and optimised.
  template <CKernelScalar T>
  [[nodiscard]] compile_result<std::unique_ptr<llvm::Module>>
  module(llvm::LLVMContext &ctx, const rt::Graph<T> &g, const Options &opt,
         const std::string &name) const {
    return target().and_then([&](std::unique_ptr<llvm::TargetMachine> tm)
                                 -> compile_result<std::unique_ptr<llvm::Module>> {
      quiet_vectoriser(ctx);
      auto m = detail::emit_module(ctx, g, name, lanes<T>(opt, g.has_tensors()),
                                   jit->getDataLayout(),
                                   tm->getTargetTriple().str());
      if (!m) {
        // The emitter has already written the verifier's own diagnosis.
        return std::unexpected{compile_error{
            errc::jit_verify, "the emitted module failed verification"}};
      }
      optimize(*m, *tm, opt, libmvec);
      return m;
    });
  }

  // At the codegen level for the backend, and at the default for the IR
  // pipeline, whose cost model reads the machine's level.
  [[nodiscard]] compile_result<std::unique_ptr<llvm::TargetMachine>>
  target(const std::optional<OptLevel> codegen = {}) const {
    auto recipe = *machine;
    if (codegen) {
      // OptLevel is LLVM's own 0..3 scale.
      recipe.setCodeGenOptLevel(
          llvm::CodeGenOpt::getLevel(std::to_underlying(*codegen))
              .value_or(llvm::CodeGenOptLevel::Default));
    }
    auto tm = recipe.createTargetMachine();
    if (!tm) {
      return std::unexpected{as_error(
          errc::jit_target, "creating a target machine", tm.takeError())};
    }
    return std::move(*tm);
  }

  template <CKernelScalar T>
  [[nodiscard]] static compile_result<Kernel<T>>
  compile(const std::shared_ptr<Impl> &self, const rt::Graph<T> &g,
          const Options &opt) {
    const std::string name = "ddx_kernel_" + std::to_string(self->counter++);
    llvm::LLVMContext ctx;
    return self->module(ctx, g, opt, name)
        .and_then([&](std::unique_ptr<llvm::Module> m) {
          return self->target(opt.codegen.codegen_level)
              .and_then([&](std::unique_ptr<llvm::TargetMachine> tm)
                            -> compile_result<std::unique_ptr<llvm::MemoryBuffer>> {
                auto object = llvm::orc::SimpleCompiler{*tm}(*m);
                if (!object) {
                  return std::unexpected{as_error(errc::jit_module, "compiling",
                                                  object.takeError())};
                }
                return std::move(*object);
              });
        })
        .and_then([&](std::unique_ptr<llvm::MemoryBuffer> object)
                      -> compile_result<Kernel<T>> {
          if (auto e = self->jit->addObjectFile(std::move(object))) {
            return std::unexpected{
                as_error(errc::jit_module, "adding the object", std::move(e))};
          }
          auto sym = self->jit->lookup(name);
          if (!sym) {
            return std::unexpected{as_error(errc::jit_lookup,
                                            llvm::Twine{"looking up "} + name,
                                            sym.takeError())};
          }
          return Kernel<T>{
              sym->toPtr<typename Kernel<T>::function_type>(),
              KernelShape::of(g, self->lanes<T>(opt, g.has_tensors())),
              externs_of(g), self};
        });
  }
};

Compiler::Compiler(std::shared_ptr<Impl> impl) noexcept
    : impl_{std::move(impl)} {}

compile_result<Compiler> Compiler::create() {
  return Impl::bring_up().transform(
      [](std::shared_ptr<Impl> impl) { return Compiler{std::move(impl)}; });
}

template <CKernelScalar T>
compile_result<Kernel<T>> Compiler::compile(const rt::Graph<T> &g,
                                            const Options &opt) {
  return Impl::compile(impl_, g, opt);
}

template <CKernelScalar T>
compile_result<std::string> Compiler::ir(const rt::Graph<T> &g,
                                         const Options &opt) const {
  llvm::LLVMContext ctx;
  return impl_->module(ctx, g, opt, "ddx_kernel_dump")
      .transform([](std::unique_ptr<llvm::Module> m) {
        std::string out;
        llvm::raw_string_ostream os(out);
        m->print(os, nullptr);
        return out;
      });
}

#define DDX_JIT_INSTANCES(T)                                                   \
  template DDX_INSTANCE_API compile_result<Kernel<T>> Compiler::compile(       \
      const rt::Graph<T> &, const Options &);                                  \
  template DDX_INSTANCE_API compile_result<std::string> Compiler::ir(           \
      const rt::Graph<T> &, const Options &) const;
DDX_JIT_INSTANCES(double)
DDX_JIT_INSTANCES(float)
#undef DDX_JIT_INSTANCES

} // namespace ddx::jit
