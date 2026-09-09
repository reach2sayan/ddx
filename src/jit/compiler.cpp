#include "codegen.hpp"

#include "rt/archive/archive.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"

#include <boost/hash2/fnv1a.hpp>
#include <boost/hash2/hash_append.hpp>
#include <boost/scope/scope_exit.hpp>

#include <llvm/Config/llvm-config.h>

#include <llvm/ADT/StringMap.h>

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/ReplaceWithVeclib.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
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
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>
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
  static std::once_flag once;
  std::call_once(once, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });
}

// The one seam between LLVM's error model and ours.
[[nodiscard]] error as_error(const errc code, const llvm::Twine &what,
                             llvm::Error e) {
  return error{.code = code,
               .detail = (what + ": " + llvm::toString(std::move(e))).str()};
}

// Defined outright: GetForCurrentProcess resolves whatever a loaded module
// exports, which need not be the libm this TU was compiled against.
[[nodiscard]] llvm::Error define_libm(llvm::orc::ExecutionSession &es,
                                      llvm::orc::JITDylib &jd,
                                      const llvm::DataLayout &dl) {
  llvm::orc::MangleAndInterner mangle(es, dl);
  llvm::orc::SymbolMap syms;
  syms.reserve(rt::op_count);
  // The unary + turns each lambda into a plain function pointer.
  const auto def = [&](const char *name, auto *fn) {
    syms[mangle(name)] = {llvm::orc::ExecutorAddr::fromPtr(fn),
                          llvm::JITSymbolFlags::Exported |
                              llvm::JITSymbolFlags::Callable};
  };
#define DDX_JIT_LIBM(fn, Op, label, ...)                                       \
  def(label, +[](double u) noexcept { return std::fn(u); });
  DDX_UNARY_MATH_TABLE(DDX_JIT_LIBM)
#undef DDX_JIT_LIBM
  def("fabs", +[](double u) noexcept { return std::fabs(u); });
  def("pow", +[](double l, double r) noexcept { return std::pow(l, r); });
  def("atan2", +[](double l, double r) noexcept { return std::atan2(l, r); });
  def("hypot", +[](double l, double r) noexcept { return std::hypot(l, r); });
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

// The widest fixed width the vector library serves a transcendental at: LLVM
// 20's libmvec table stops at four doubles, so an AVX-512 host's eight-wide
// intrinsic has no library form and unrolls to scalar calls.
[[nodiscard]] unsigned library_lanes(const llvm::Triple &triple) {
  const auto tlii = target_library_info(
      triple, Options{.codegen = {.veclib = VecLib::Libmvec}}, true);
  llvm::ElementCount fixed;
  llvm::ElementCount scalable;
  tlii.getWidestVF("sin", fixed, scalable);
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
constexpr const char *kRetainFlag = "ddx.retain_object";

[[nodiscard]] std::optional<std::uint64_t> flag_of(const llvm::Module &m,
                                                   const char *key) {
  const auto *const flag =
      llvm::mdconst::extract_or_null<llvm::ConstantInt>(m.getModuleFlag(key));
  return flag != nullptr ? std::optional{flag->getZExtValue()} : std::nullopt;
}

[[nodiscard]] bool retains_object(const llvm::Module &m) {
  return flag_of(m, kRetainFlag).value_or(0) != 0;
}

// The flag is written from std::to_underlying(Level), which is LLVM's own
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
  LevelledCompiler(llvm::orc::JITTargetMachineBuilder machine,
                   llvm::ObjectCache *objects)
      : IRCompiler(llvm::orc::irManglingOptionsFromTargetOptions(
            machine.getOptions())),
        machine_(std::move(machine)), objects_(objects) {}

  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  operator()(llvm::Module &m) override {
    auto machine = machine_;
    machine.setCodeGenOptLevel(codegen_level_of(m));
    auto tm = machine.createTargetMachine();
    // Handed over only where the compile asked to keep its object.
    return (!tm) ? tm.takeError()
                 : llvm::orc::SimpleCompiler{
                       **tm, retains_object(m) ? objects_ : nullptr}(m);
  }

private:
  llvm::orc::JITTargetMachineBuilder machine_;
  llvm::ObjectCache *objects_;
};

// The backend's output caught on its way past: notifyObjectCompiled() hands
// over the buffer the linker is about to get, so nothing is re-serialised.
// Keyed by module identifier, all the callback is given.  A hand-off, not a
// cache: an entry is taken out once, by the compile that put it in.
class Objects final : public llvm::ObjectCache {
public:
  void notifyObjectCompiled(const llvm::Module *m,
                            llvm::MemoryBufferRef obj) override {
    const auto bytes =
        std::as_bytes(std::span{obj.getBuffer().data(), obj.getBufferSize()});
    auto kept = std::make_shared<const std::vector<std::byte>>(bytes.begin(),
                                                               bytes.end());
    const std::lock_guard lock{mutex_};
    kept_[m->getModuleIdentifier()] = std::move(kept);
  }

  // The disk half is consulted before the compile layer is reached at all, and
  // nothing means "compile it".
  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *) override {
    return nullptr;
  }

  // Forget them: holding on would keep every object ever compiled alive.
  [[nodiscard]] Kernel::object_type take(llvm::StringRef id) {
    const std::lock_guard lock{mutex_};
    const auto it = kept_.find(id);
    if (it == kept_.end()) {
      return {};
    }
    auto out = std::move(it->second);
    kept_.erase(it);
    return out;
  }

private:
  std::mutex mutex_;
  // The key is a module identifier and this is LLVM's container for one: a
  // single allocation per entry with the characters inside it, where a
  // node-based map pays for a std::string of its own as well.
  llvm::StringMap<Kernel::object_type> kept_;
};

// Everything an object file must agree with, as one line.  The feature string
// is folded, not spelled: detectHost() answers with 1.5 KB of
// `+avx2,-avx512f,...` and a reader can only act on *that* they differ.
[[nodiscard]] std::string
host_identity_of(const llvm::orc::JITTargetMachineBuilder &machine) {
  const std::string features = machine.getFeatures().getString();
  boost::hash2::fnv1a_64 h;
  h.update(features.data(), features.size());
  const llvm::StringRef cpu = machine.getCPU();
  return std::format("{}/{}/feat-{:016x}/llvm-" LLVM_VERSION_STRING,
                     machine.getTargetTriple().str(),
                     cpu.empty() ? std::string_view{"generic"}
                                 : std::string_view{cpu.data(), cpu.size()},
                     h.result());
}

// Asked of the target's cost model rather than a feature list, so
// prefer-256-bit parts answer 4 as the backend would split them anyway.
[[nodiscard]] unsigned host_lanes(llvm::orc::JITTargetMachineBuilder machine) {
  auto tm = machine.createTargetMachine();
  if (!tm) {
    llvm::consumeError(tm.takeError());
    return 1;
  }
  llvm::LLVMContext ctx;
  llvm::Module probe("ddx.probe", ctx);
  auto *const fn = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false),
      llvm::Function::ExternalLinkage, "probe", probe);
  const llvm::TypeSize bits =
      (*tm)->getTargetTransformInfo(*fn).getRegisterBitWidth(
          llvm::TargetTransformInfo::RGK_FixedWidthVector);
  return std::max(1u, static_cast<unsigned>(bits.getFixedValue() / 64));
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
  pto.SLPVectorization = opt.codegen.slp;
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
  Objects &objects;
  std::shared_ptr<void> code;

  [[nodiscard]] unsigned lanes_for(const Options &opt) const {
    return emitted_lanes(opt, triple, libmvec, lanes, veclib_lanes);
  }
};

// One graph compiled once, filling the report it was handed as it goes.
class Compilation : private impl::pinned {
public:
  Compilation(Host host, const rt::Graph<double> &g, const Options &opt,
              std::string name, CompileReport &into)
      : host_(std::move(host)), g_(g), opt_(opt), name_(std::move(name)),
        rep_(into) {}

  [[nodiscard]] result<llvm::orc::ThreadSafeModule> prepared() {
    return emitted().and_then([this](llvm::orc::ThreadSafeModule m) {
      return optimized(std::move(m));
    });
  }

  [[nodiscard]] result<Kernel> operator()() {
    return prepared().and_then([this](llvm::orc::ThreadSafeModule m) {
      return materialised(std::move(m));
    });
  }

private:
  [[nodiscard]] result<llvm::orc::ThreadSafeModule> emitted() {
    const boost::scope::scope_exit clock{
        [this, start = Clock::now()] { rep_.emit = Clock::now() - start; }};
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto m = detail::emit_module(*ctx, g_, name_, host_.lanes_for(opt_),
                                 host_.jit.getDataLayout(), host_.triple);
    if (!m) {
      // The emitter has already written the verifier's own diagnosis.
      return std::unexpected{
          error{errc::jit_verify, "the emitted module failed verification"}};
    }
    // The object cache keys on the module and LLVM prints this in diagnostics;
    // the kernel's name is already unique per compile.
    m->setModuleIdentifier(name_);
    m->addModuleFlag(llvm::Module::Error, kCodegenFlag,
                     std::to_underlying(opt_.codegen.codegen_level));
    m->addModuleFlag(llvm::Module::Error, kRetainFlag,
                     static_cast<unsigned>(opt_.retain_object));
    rep_.nodes = g_.live_count();
    rep_.instructions = m->getInstructionCount();
    return llvm::orc::ThreadSafeModule{
        std::move(m), llvm::orc::ThreadSafeContext{std::move(ctx)}};
  }

  [[nodiscard]] result<llvm::orc::ThreadSafeModule>
  optimized(llvm::orc::ThreadSafeModule m) {
    const boost::scope::scope_exit clock{
        [&, start = Clock::now()] { rep_.optimize = Clock::now() - start; }};
    // One machine per compile -- see Host.
    auto tm = host_.machine.createTargetMachine();
    if (!tm) {
      return std::unexpected{as_error(
          errc::jit_target, "creating a target machine", tm.takeError())};
    }
    std::optional<error> refused;
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
  [[nodiscard]] result<Kernel> materialised(llvm::orc::ThreadSafeModule m) {
    const boost::scope::scope_exit clock{
        [this, start = Clock::now()] { rep_.codegen = Clock::now() - start; }};
    if (auto e = host_.jit.addIRModule(std::move(m))) {
      return std::unexpected{
          as_error(errc::jit_module, "adding the module", std::move(e))};
    }
    auto sym = host_.jit.lookup(name_);
    if (!sym) {
      return std::unexpected{
          as_error(errc::jit_lookup, llvm::Twine{"looking up "} + name_,
                   sym.takeError())};
    }
    return Kernel{sym->toPtr<Kernel::function_type>(), KernelShape::of(g_),
                  std::move(host_.code),
                  std::make_shared<const std::string>(name_),
                  host_.objects.take(name_)};
  }

  Host host_;
  const rt::Graph<double> &g_;
  const Options &opt_;
  std::string name_;
  CompileReport &rep_;
};

// --- objects kept between runs -----------------------------------------------
//
// Not compiling the same graph twice dominates every other optimisation here: a
// compile costs seconds against a graph's microseconds.
//
// The lookup does *not* go through llvm::ObjectCache, whose hook fires after
// emission and the pass pipeline and so would skip codegen alone; keyed on the
// graph, the whole compile is skipped.  Content-addressed, so a stale entry is
// a miss, and nothing here fails a compile -- every failure is "compile it".
constexpr std::string_view kCacheMagic = "ddxjitob";
constexpr std::uint32_t kCacheFormat = 2;

// Bumped by hand for what neither the digest nor the entry's shape can see: the
// emitter, the pass pipeline, the ABI.  Folded into the key, so a bump misses
// rather than refuses.
constexpr std::uint32_t kCacheEpoch = 1;

// The identity, with the width *emitted* in place of the one asked for: a
// derived `Lanes` means the host's, so the raw request would give one graph two
// keys that never hit each other.
[[nodiscard]] std::uint64_t
cache_key(std::uint64_t graph, std::string_view host, const Codegen &emitted) {
  boost::hash2::fnv1a_64 h;
  boost::hash2::hash_append(h, rt::detail::wire_flavor{}, host);
  rt::detail::fold(h, graph);
  rt::detail::fold(h, kCacheEpoch);
  rt::detail::fold(h, emitted);
  return h.result();
}

[[nodiscard]] std::filesystem::path entry_path(std::string_view dir,
                                               std::uint64_t key) {
  return std::filesystem::path{dir} / std::format("{:016x}.ddxjit", key);
}

// Chosen by the compile that produced the bytes, and underivable, so it travels
// with them.  Described, so the same codec that writes a graph writes this: the
// length prefix and its bounds are the archive's, not a second hand-rolled
// pair.
struct Entry {
  std::string symbol;
  std::vector<std::byte> code;
};
BOOST_DESCRIBE_STRUCT(Entry, (), (symbol, code))

// The prologue's shape and the entry's, so a field added to either refuses the
// old entries by itself.
constexpr std::uint32_t kCacheSchema = rt::detail::Container::stamp<Entry>();

[[nodiscard]] std::optional<Entry> read_entry(std::string_view dir,
                                              std::uint64_t key) {
  auto bytes = rt::detail::Container::read(entry_path(dir, key));
  if (!bytes) {
    return std::nullopt; // absent, unreadable: both are a miss
  }
  // unpack has cleared the checksum before this sees a payload byte.  Every
  // prologue field written is checked and none is written that is not: the
  // prologue is the one part no checksum covers.  `model_nodes` belongs to the
  // graph half of the format and is written as zero.
  const auto opened = rt::detail::Container::unpack(*bytes, kCacheMagic);
  if (!opened) {
    return std::nullopt;
  }
  const auto &[head, payload] = *opened;
  if (head.format != kCacheFormat || head.schema != kCacheSchema ||
      head.scalar_size != sizeof(double) ||
      head.scalar_kind != rt::detail::ScalarKind::Floating ||
      head.model_nodes != 0 || head.model_digest != key) {
    return std::nullopt;
  }
  // A forged length is the codec's problem now: it bounds every one against the
  // bytes actually left, so the 4 + symbol_bytes wrap this used to widen for
  // cannot be spelled.
  auto entry = rt::detail::Container::decode<Entry>(payload);
  if (!entry || entry->symbol.empty() || entry->code.empty()) {
    return std::nullopt;
  }
  return std::move(*entry);
}

void write_entry(std::string_view dir, std::uint64_t key,
                 std::string_view symbol, std::span<const std::byte> code) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    return; // A cache that cannot be written is a cache that always misses.
  }
  const Entry entry{.symbol = std::string{symbol},
                    .code = {code.begin(), code.end()}};
  const rt::detail::FileHeader h{.magic = kCacheMagic,
                                 .format = kCacheFormat,
                                 .schema = kCacheSchema,
                                 .scalar_size = sizeof(double),
                                 .scalar_kind =
                                     rt::detail::ScalarKind::Floating,
                                 .model_digest = key};
  // Staged inside the cache directory: a rename is only atomic within one
  // filesystem.
  (void)rt::detail::Container::write(
      entry_path(dir, key),
      rt::detail::Container::pack(kCacheMagic, h,
                                  rt::detail::Container::encode(entry)));
}

// Where a background compile runs.  Not a member of Impl: a Kernel holds a
// share of the Impl it came from, so the last share can be dropped on a worker,
// and a pool joined from its own thread deadlocks.
//
// `warm_` is a member so the order is a class invariant.  LLVM registers atexit
// entries lazily *while it compiles*, and anything registered after this object
// is destroyed before it -- so the pool's join would run workers against freed
// state.  Compiling inside a member that completes first puts LLVM's entries
// strictly below the pool's.  Templated only to avoid naming the private Impl.
template <typename I>
  requires requires(const std::shared_ptr<I> &impl, const rt::Graph<double> &g,
                    CompileReport &rep) {
    { I::run(impl, g, Options{}, rep) } -> std::same_as<result<Kernel>>;
  }
class Compiles : private impl::pinned {
public:
  // Brought up once, and never torn down before the statics its workers touch.
  [[nodiscard]] static Compiles &shared(const std::shared_ptr<I> &impl) {
    static Compiles instance{impl};
    return instance;
  }

  [[nodiscard]] llvm::DefaultThreadPool &threads() noexcept { return threads_; }

private:
  explicit Compiles(const std::shared_ptr<I> &impl) : warm_(impl) {}

  // Constructing one compiles something, which is what makes LLVM register;
  // any live Impl will do.
  struct Warm {
    explicit Warm(const std::shared_ptr<I> &impl) {
      rt::Builder<double> b;
      const auto x = rt::var(b, "x");
      const auto y = rt::var(b, "y");
      // Off the same table the emitter emits from: what LLVM touches lazily
      // follows the ops, so enumerate them.
      auto e = x / (y + 1.0) - x * y;
#define DDX_JIT_WARM(fn, Op, label, ...) e += fn(x);
      DDX_UNARY_MATH_TABLE(DDX_JIT_WARM)
#undef DDX_JIT_WARM
      e += pow(x, y) + atan2(x, y) + hypot(x, y);
      e += abs(x) + max(x, y) + min(x, y) + sign(x) + (-x);
      // Jacobian too: reverse mode emits ops the value alone never reaches.
      const auto g =
          rt::GraphBuilder<double>{b}.value(e).build_jacobian().finish();
      CompileReport discard;
      (void)I::run(impl, g, Options{}, discard);
    }
  };

  Warm warm_; // LLVM registers while this runs, so before we do
  llvm::DefaultThreadPool threads_;
};

} // namespace

struct Compiler::Impl {
  // Before `jit` and it must stay there: the compile layer holds a pointer to
  // this, and members are destroyed in reverse declaration order.
  Objects objects;
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
  bool libmvec = false;      // Whether the vector forms resolve here
  unsigned lanes = 1;        // The host's vector width in doubles
  unsigned veclib_lanes = 1; // The widest the vector library serves

  [[nodiscard]] static std::expected<std::shared_ptr<Impl>, error> bring_up() {
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
    impl->lanes = host_lanes(*jtmb);
    impl->veclib_lanes = library_lanes(jtmb->getTargetTriple());
    impl->host = host_identity_of(*jtmb);

    // Not setNumCompileThreads: the backend runs on the thread that asked, and
    // no idle pool is spawned.
    auto compiler_factory =
        [objects = &impl->objects](llvm::orc::JITTargetMachineBuilder machine)
        -> llvm::Expected<
            std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
      return std::make_unique<LevelledCompiler>(std::move(machine), objects);
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
    if (auto e = define_libm(impl->jit->getExecutionSession(), jd, dl)) {
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

  // One per adopted object: the symbol is baked into the bytes, so two objects
  // from one graph share a name and the main dylib would refuse the second.
  // The link order keeps libm and the vector-math generators reachable.
  [[nodiscard]] static result<Kernel> link(const std::shared_ptr<Impl> &self,
                                           std::span<const std::byte> object,
                                           std::string_view symbol,
                                           KernelShape shape) {
    auto &jit = *self->jit;
    auto jd =
        jit.createJITDylib("ddx_adopted_" + std::to_string(self->counter++));
    if (!jd) {
      return std::unexpected{
          as_error(errc::jit_module, "creating a dylib", jd.takeError())};
    }
    jd->addToLinkOrder(jit.getMainJITDylib());

    // Copied: the linker keeps the buffer as long as the code lives.  Kept as
    // well, so an adopted kernel saves again like any other.
    auto kept = std::make_shared<const std::vector<std::byte>>(object.begin(),
                                                               object.end());
    if (auto e = jit.addObjectFile(
            *jd, llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef{
                     reinterpret_cast<const char *>(object.data()),
                     object.size()}))) {
      return std::unexpected{
          as_error(errc::jit_object, "adding the object", std::move(e))};
    }

    // Where a malformed object is found: addObjectFile only queues it, and the
    // link runs on the lookup.
    auto sym = jit.lookup(*jd, symbol);
    if (!sym) {
      return std::unexpected{as_error(errc::jit_lookup,
                                      llvm::Twine{"looking up "} +
                                          llvm::StringRef{symbol},
                                      sym.takeError())};
    }
    return Kernel{sym->toPtr<Kernel::function_type>(), shape, self,
                  std::make_shared<const std::string>(symbol), std::move(kept)};
  }

  // On Impl, so a queued compile needs only a share of this and not a Compiler
  // that may have been moved from.
  // What a Compilation is handed: the process-wide JIT, and a share of this,
  // so the code outlives any Compiler that goes away mid-compile.
  [[nodiscard]] static Host host_of(const std::shared_ptr<Impl> &self) {
    return Host{.jit = *self->jit,
                .machine = *self->machine,
                .triple = self->triple,
                .libmvec = self->libmvec,
                .lanes = self->lanes,
                .veclib_lanes = self->veclib_lanes,
                .objects = self->objects,
                .code = self};
  }

  [[nodiscard]] static result<Kernel> run(const std::shared_ptr<Impl> &self,
                                          const rt::Graph<double> &g,
                                          const Options &opt,
                                          CompileReport &rep) {
    Host host = host_of(self);
    const bool caching = !opt.cache_dir.empty();
    Codegen emitted = opt.codegen;
    emitted.lanes = *Lanes::exactly(host.lanes_for(opt));
    const std::uint64_t key =
        caching ? cache_key(rt::digest(g), self->host, emitted) : 0;
    if (caching) {
      if (const auto entry = read_entry(opt.cache_dir, key)) {
        // The shape comes from the live graph, never the file: a forged entry
        // supplies code and a symbol, never a column count.
        if (auto adopted =
                link(self, entry->code, entry->symbol, KernelShape::of(g));
            adopted) {
          // The phases are zero because they did not happen.
          rep.nodes = g.live_count();
          return adopted;
        }
        // A miss like any other, left on disk: another process may be
        // mid-write, and a stale entry costs one failed link.
      }
    }

    Options effective = opt;
    // A compile whose object will be stored has to keep it, whatever was asked.
    effective.retain_object = effective.retain_object || caching;

    Compilation work{std::move(host), g, effective,
                     "ddx_kernel_" + std::to_string(self->counter++), rep};
    auto kernel = work();
    if (caching && kernel && !kernel->object().empty()) {
      write_entry(opt.cache_dir, key, kernel->symbol(), kernel->object());
    }
    return kernel;
  }
};

Compiler::Compiler(std::shared_ptr<Impl> impl) noexcept
    : impl_{std::move(impl)} {}

std::string_view Compiler::host_identity() const noexcept {
  return impl_->host;
}

result<Compiler> Compiler::create() {
  return Impl::bring_up().transform(
      [](std::shared_ptr<Impl> impl) { return Compiler{std::move(impl)}; });
}
Compiler::~Compiler() = default;
Compiler::Compiler(Compiler &&) noexcept = default;
Compiler &Compiler::operator=(Compiler &&) noexcept = default;

result<Kernel> Compiler::compile(const rt::Graph<double> &g, const Options &opt,
                                 CompileReport *report) {
  CompileReport discard;
  return Impl::run(impl_, g, opt, report != nullptr ? *report : discard);
}

// packaged_task, never std::async: only the latter's future joins in its
// destructor, putting a compile on the critical path of whoever dropped it.
std::shared_future<result<Kernel>>
Compiler::compile_async(std::shared_ptr<const rt::Graph<double>> g,
                        Options opt) {
  auto task = std::make_shared<std::packaged_task<result<Kernel>()>>(
      [self = impl_, graph = std::move(g), opt] {
        CompileReport discard;
        return Impl::run(self, *graph, opt, discard);
      });
  auto landing = task->get_future().share();
  // The task holds the future as well as the promise, so a caller dropping its
  // copy never tears the shared state down under a running compile.  The
  // shared_ptr is because std::function must be copyable.
  Compiles<Impl>::shared(impl_).threads().async(
      [task, landing] { std::invoke(*task); });
  return landing;
}

result<Kernel> Compiler::adopt(std::span<const std::byte> object,
                               std::string_view symbol, KernelShape shape) {
  return Impl::link(impl_, object, symbol, shape);
}

result<std::string> Compiler::render_ir(const rt::Graph<double> &g,
                                        const Options &opt) const {
  CompileReport discard;
  Compilation run{Impl::host_of(impl_), g, opt, "ddx_kernel_dump", discard};
  return run.prepared().transform([](llvm::orc::ThreadSafeModule m) {
    std::string out;
    llvm::raw_string_ostream os(out);
    m.withModuleDo([&](llvm::Module &mod) { mod.print(os, nullptr); });
    return out;
  });
}

} // namespace ddx::jit
