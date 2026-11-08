#include "codegen.hpp"

#include "spelling.hpp"

#include "rt/interpret.hpp"
#include "rt/pipeline.hpp"
#include "util/meta.hpp"

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

// Target initialisation is global LLVM state, and a process may hold more than
// one Compiler, so it runs once.
void init_native_target_once() {
  [[maybe_unused]] static const bool once = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
}

// Converts an llvm::Error into a compile_error.
[[nodiscard]] compile_error as_error(const errc code, const llvm::Twine &what,
                                     llvm::Error e) {
  return compile_error{.code = code,
                       .detail =
                           (what + ": " + llvm::toString(std::move(e))).str()};
}

// Converts an llvm::Expected or llvm::Error into a compile_result, so that
// DDX_TRY and DDX_CHECK can consume LLVM results.
template <typename X>
[[nodiscard]] compile_result<X> lifted(llvm::Expected<X> x, const errc code,
                                       const llvm::Twine &what) {
  if (!x) {
    return std::unexpected{as_error(code, what, x.takeError())};
  }
  return std::move(*x);
}
[[nodiscard]] compile_result<void> lifted(llvm::Error e, const errc code,
                                          const llvm::Twine &what) {
  if (e) {
    return std::unexpected{as_error(code, what, std::move(e))};
  }
  return {};
}

// Defines the runtime symbol for a Call opcode using the opcode's functor.
// Opcodes of any other form define nothing.
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

// Tensor kernels use a power-of-two width up to this limit, matching the
// widths the interpreter's tensor step is instantiated at.
constexpr std::size_t tensor_widths = 5;
constexpr unsigned widest_tensor_lanes = 1u << (tensor_widths - 1);

// Runs the interpreter's step for node v on the kernel's tapes, at the
// kernel's width.
template <CKernelScalar T>
void tensor_step(const rt::detail::TensorCall<T> &c, rt::NodeId v,
                 std::size_t lanes, T *tape, T *vtape) {
  const rt::Graph<T> &g = *c.graph;
  const auto t = rt::detail::TapeView<T>::of(g, vtape, c.kernels);
  meta::with_index<tensor_widths>(
      static_cast<std::size_t>(std::countr_zero(lanes)), [&](auto i) {
        constexpr std::size_t W = std::size_t{1} << i;
        const auto lane =
            rt::detail::lanes_of<W>(std::span<T>{tape, g.tape_rows() * W});
        rt::detail::lanes_tensor<W>(*c.arena, v, (*c.arena)[v],
                                    lane(g.rows()[v]), lane, t);
      });
}

// Scatters every output block for the first `live` of the kernel's lanes.
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

// Defines every runtime symbol a kernel over T calls, such as sin or sinf.
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

// Defines every non-intrinsic callee explicitly, because the process symbol
// search could resolve to a different libm than this file was compiled with.
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

// `have` says whether libmvec loaded in this process; promising a vector sin
// that cannot be resolved fails at link time.
[[nodiscard]] bool want_veclib(const Options &opt, const bool have) {
  return have && opt.codegen.veclib == VecLib::Libmvec;
}

llvm::TargetLibraryInfoImpl target_library_info(const llvm::Triple &triple,
                                                const Options &opt,
                                                const bool have) {
  llvm::TargetLibraryInfoImpl tlii(triple);
  if (want_veclib(opt, have)) {
    tlii.addVectorizableFunctionsFromVecLib(
        llvm::TargetLibraryInfoImpl::LIBMVEC_X86, triple);
  }
  return tlii;
}

// The widest vector width libmvec provides for T. LLVM 20 stops at four
// doubles, so wider AVX-512 calls would fall back to scalar libm.
template <CKernelScalar T>
[[nodiscard]] unsigned library_lanes(const llvm::Triple &triple) {
  const auto tlii = target_library_info(
      triple, Options{.codegen = {.veclib = VecLib::Libmvec}}, true);
  llvm::ElementCount fixed;
  llvm::ElementCount scalable;
  tlii.getWidestVF(detail::libm_name<T>("sin"), fixed, scalable);
  return std::max(1u, static_cast<unsigned>(fixed.getFixedValue()));
}

// Asks the target's cost model rather than a feature list, so CPUs that
// prefer 256-bit vectors report 256.
[[nodiscard]] unsigned vector_bits(llvm::orc::JITTargetMachineBuilder machine) {
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

// Runs the optimisation pipeline. With libmvec, vector intrinsics are rewritten
// to its entry points; otherwise the backend unrolls them into scalar calls.
void optimize(llvm::Module &m, llvm::TargetMachine &tm, const Options &opt,
              const bool have_libmvec) {
  const llvm::Triple &triple = tm.getTargetTriple();
  llvm::TargetLibraryInfoImpl tlii =
      target_library_info(triple, opt, have_libmvec);

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  // The point loop is already vectorised, so the loop vectoriser runs only on
  // tensor loops whose metadata forces it.
  llvm::PipelineTuningOptions pto;
  pto.SLPVectorization = opt.codegen.slp_vectorize;
  pto.LoopVectorization = false;

  llvm::PassBuilder pb(&tm, pto);
  // Must precede registerFunctionAnalyses, which would otherwise install a
  // TargetLibraryAnalysis with an empty vector-function table.
  fam.registerPass([&] { return llvm::TargetLibraryAnalysis(tlii); });
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  static const std::array levels{
      llvm::OptimizationLevel::O0, llvm::OptimizationLevel::O1,
      llvm::OptimizationLevel::O2, llvm::OptimizationLevel::O3};
  const llvm::OptimizationLevel level =
      levels[std::to_underlying(opt.codegen.opt_level)];

  llvm::ModulePassManager mpm = level == llvm::OptimizationLevel::O0
                                    ? pb.buildO0DefaultPipeline(level)
                                    : pb.buildPerModuleDefaultPipeline(level);
  if (want_veclib(opt, have_libmvec)) {
    llvm::FunctionPassManager fpm;
    fpm.addPass(llvm::ReplaceWithVeclib());
    mpm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
  }
  mpm.run(m, mam);
}

// Silences the vectoriser's stderr notes when it declines a tensor loop, since
// the loop then simply runs scalar.
void quiet_vectoriser(llvm::LLVMContext &ctx) {
  ctx.setDiagnosticHandlerCallBack(
      [](const llvm::DiagnosticInfo *d, void *) {
        if (d->getKind() == llvm::DK_OptimizationFailure) {
          return;
        }
        llvm::DiagnosticPrinterRawOStream out(llvm::errs());
        llvm::errs() << llvm::LLVMContext::getDiagnosticMessagePrefix(
                            d->getSeverity())
                     << ": ";
        d->print(out);
        llvm::errs() << '\n';
      },
      nullptr, /*RespectFilters=*/true);
}

} // namespace

// Each Kernel holds a shared_ptr to Impl, so its code outlives the Compiler.
struct Compiler::Impl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  // A builder rather than a TargetMachine, which caches subtargets in an
  // unsynchronised map; each compile creates its own machine.
  std::optional<llvm::orc::JITTargetMachineBuilder> machine;
  // Gives each kernel a unique name, so that concurrent compiles never define
  // the same symbol twice.
  std::atomic<unsigned> counter{0};
  bool libmvec = false; // Whether libmvec loaded in this process
  unsigned bits = 64;   // The host's vector register width in bits
  // The widest libmvec width for double, then for float.
  std::array<unsigned, 2> veclib_lanes{1, 1};

  [[nodiscard]] static compile_result<std::shared_ptr<Impl>> bring_up() {
    init_native_target_once();
    auto impl = std::make_shared<Impl>();

    // Targets the host CPU, matching the project's -march=native.
    DDX_TRY(jtmb, lifted(llvm::orc::JITTargetMachineBuilder::detectHost(),
                         errc::jit_target, "detecting the host"));
    impl->bits = vector_bits(*jtmb);
    impl->veclib_lanes = {library_lanes<double>(jtmb->getTargetTriple()),
                          library_lanes<float>(jtmb->getTargetTriple())};

    // Copy the builder after prepareForConstruction sets JITLink's small code
    // model; the default large model costs a 64-bit load per constant.
    llvm::orc::LLJITBuilder builder;
    builder.setJITTargetMachineBuilder(std::move(*jtmb));
    DDX_CHECK(lifted(builder.prepareForConstruction(), errc::jit_target,
                     "preparing the JIT"));
    impl->machine = *builder.JTMB;
    DDX_TRY(jit,
            lifted(builder.create(), errc::jit_target, "creating the JIT"));
    impl->jit = std::move(*jit);

    llvm::orc::JITDylib &jd = impl->jit->getMainJITDylib();
    const llvm::DataLayout &dl = impl->jit->getDataLayout();
    const char prefix = dl.getGlobalPrefix();

    // Defined before the generators are added, so these definitions win.
    DDX_CHECK(lifted(define_runtime(impl->jit->getExecutionSession(), jd, dl),
                     errc::jit_target, "defining libm"));

    DDX_TRY(
        procs,
        lifted(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                   prefix),
               errc::jit_target, "opening the process symbols"));
    jd.addGenerator(std::move(*procs));

    // libmvec is loaded explicitly, since the process search cannot see it
    // unless the program already uses it.
    if (auto vec = llvm::orc::DynamicLibrarySearchGenerator::Load(
            "libmvec.so.1", prefix)) {
      jd.addGenerator(std::move(*vec));
      impl->libmvec = true;
    } else {
      // Without libmvec the transcendentals simply stay scalar.
      llvm::consumeError(vec.takeError());
    }
    return impl;
  }

  // A stated width is used as is; otherwise the host width, capped by libmvec.
  // Tensor graphs round down to a width the tensor step is instantiated at.
  template <CKernelScalar T>
  [[nodiscard]] unsigned lanes(const Options &opt, const bool tensors) const {
    const unsigned host = bits / unsigned{8 * sizeof(T)};
    const unsigned w = opt.codegen.lanes.stated().value_or(
        want_veclib(opt, libmvec)
            ? std::min(host, veclib_lanes[std::same_as<T, float>])
            : host);
    return tensors ? std::bit_floor(std::min(w, widest_tensor_lanes)) : w;
  }

  template <CKernelScalar T>
  [[nodiscard]] compile_result<std::unique_ptr<llvm::Module>>
  module(llvm::LLVMContext &ctx, const rt::Graph<T> &g, const Options &opt,
         const std::string &name) const {
    DDX_TRY(tm, target());
    quiet_vectoriser(ctx);
    auto m = detail::emit_module(ctx, g, name, lanes<T>(opt, g.has_tensors()),
                                 jit->getDataLayout(),
                                 (*tm)->getTargetTriple().str());
    if (!m) {
      // The verifier has already printed its diagnosis.
      return std::unexpected{
          compile_error{.code = errc::jit_verify,
                        .detail = "the emitted module failed verification"}};
    }
    optimize(*m, **tm, opt, libmvec);
    return m;
  }

  // The backend uses the requested codegen level; the IR pipeline uses the
  // default, since its cost model reads the machine's level.
  [[nodiscard]] compile_result<std::unique_ptr<llvm::TargetMachine>>
  target(const std::optional<OptLevel> codegen = {}) const {
    auto recipe = *machine;
    if (codegen) {
      // OptLevel uses LLVM's own 0 to 3 scale.
      recipe.setCodeGenOptLevel(
          llvm::CodeGenOpt::getLevel(std::to_underlying(*codegen))
              .value_or(llvm::CodeGenOptLevel::Default));
    }
    return lifted(recipe.createTargetMachine(), errc::jit_target,
                  "creating a target machine");
  }

  template <CKernelScalar T>
  [[nodiscard]] static compile_result<Kernel<T>>
  compile(const std::shared_ptr<Impl> &self, const rt::Graph<T> &g,
          const Options &opt) {
    const std::string name = "ddx_kernel_" + std::to_string(self->counter++);
    llvm::LLVMContext ctx;
    DDX_TRY(module, self->module(ctx, g, opt, name));
    DDX_TRY(tm, self->target(opt.codegen.codegen_level));
    DDX_TRY(object, lifted(llvm::orc::SimpleCompiler{**tm}(**module),
                           errc::jit_module, "compiling"));
    DDX_CHECK(lifted(self->jit->addObjectFile(std::move(*object)),
                     errc::jit_module, "adding the object"));
    DDX_TRY(sym, lifted(self->jit->lookup(name), errc::jit_lookup,
                        llvm::Twine{"looking up "} + name));
    return Kernel<T>{sym->toPtr<typename Kernel<T>::function_type>(),
                     KernelShape::of(g, self->lanes<T>(opt, g.has_tensors())),
                     externs_of(g), self};
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
  template DDX_INSTANCE_API compile_result<std::string> Compiler::ir(          \
      const rt::Graph<T> &, const Options &) const;
DDX_JIT_INSTANCES(double)
DDX_JIT_INSTANCES(float)
#undef DDX_JIT_INSTANCES

} // namespace ddx::jit
