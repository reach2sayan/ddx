#include "codegen.hpp"

#include "rt/lowering.hpp"
#include "util/ranges.hpp"

#include <boost/container/small_vector.hpp>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Alignment.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace ddx::jit::detail {
namespace {

// Which intrinsic covers a Call row; the rest go out as libm calls by the
// same name.  One array indexed by the enumerator, filled once, the shape
// rt::op_info already has: lookupIntrinsicID binary-searches a name table off
// a Twine that has to be flattened into a heap string, and the emitter asks
// per node.
llvm::Intrinsic::ID intrinsic_for(rt::OpCode op) {
  static const std::array<llvm::Intrinsic::ID, rt::op_count> table = [] {
    std::array<llvm::Intrinsic::ID, rt::op_count> t{};
    for (const auto i : std::views::iota(0uz, rt::op_count)) {
      const auto &[how, name] = rt::lowering_of(static_cast<rt::OpCode>(i));
      llvm::SmallString<32> spelled{"llvm."};
      spelled += name;
      // Not pow: the library-call simplifier rewrites llvm.pow with a constant
      // operand -- 1/x for -1, sqrt for 0.5 -- and libm's pow is not
      // correctly rounded, so the kernel would leave the sweep by an ulp.
      t[i] = how == rt::Lower::Call && i != std::to_underlying(rt::OpCode::Pow)
                 ? llvm::Intrinsic::lookupIntrinsicID(spelled)
                 : llvm::Intrinsic::not_intrinsic;
    }
    return t;
  }();
  return table[std::to_underlying(op)];
}

// The IR spelling of -fno-math-errno: without it a call is assumed to write
// errno, which blocks hoisting across it.
llvm::Function *libm_decl(llvm::Module &m, std::string_view name,
                          unsigned args) {
  llvm::Type *f64 = llvm::Type::getDoubleTy(m.getContext());
  llvm::SmallVector<llvm::Type *, 2> params(args, f64);
  llvm::FunctionCallee c =
      m.getOrInsertFunction(name, llvm::FunctionType::get(f64, params, false));
  auto *fn = llvm::cast<llvm::Function>(c.getCallee());
  fn->setMemoryEffects(llvm::MemoryEffects::none());
  fn->setDoesNotThrow();
  fn->setWillReturn();
  fn->setDoesNotFreeMemory();
  return fn;
}

// The one way out of a kernel into a function the program supplied, defined
// by the runtime beside libm: `ddx_call_extern(fn, in, arity, answer,
// components)`.  The function itself comes off the kernel's extern table, so
// the object holds no address.
llvm::Function *trampoline_decl(llvm::Module &m) {
  llvm::LLVMContext &ctx = m.getContext();
  llvm::PointerType *const ptr = llvm::PointerType::getUnqual(ctx);
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::FunctionCallee c = m.getOrInsertFunction(
      trampoline_symbol,
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                              {ptr, ptr, i64, ptr, i64}, false));
  auto *fn = llvm::cast<llvm::Function>(c.getCallee());
  fn->setDoesNotThrow();
  fn->setWillReturn();
  return fn;
}

// W points per iteration, as a double at W == 1 and <W x double> otherwise.
// Every IRBuilder operation is typed on `ty`, so only the memory accesses and
// the calls with no vector intrinsic tell the two apart.
struct LaneType {
  unsigned width;
  llvm::Type *ty;
  [[nodiscard]] constexpr bool vector() const noexcept { return width > 1; }
};

class Emitter {
public:
  Emitter(llvm::Module &m, llvm::IRBuilder<> &b, LaneType lanes,
          llvm::Value *externs)
      : m_{m}, b_{b}, lanes_{lanes}, externs_{externs} {}

  // A splat where the lane type is a vector.
  [[nodiscard]] llvm::Value *constant(double v) const {
    return llvm::ConstantFP::get(lanes_.ty, v);
  }

  // Columns are plain double*, so no alignment past the element's is claimed.
  // A load past the batch reads inactive lanes as 1.0: never stored, and inside
  // every op's domain, so a scalarised libm call cannot hit a pole.
  [[nodiscard]] llvm::Value *load(llvm::Value *column, llvm::Value *index,
                                  llvm::Value *mask,
                                  const llvm::Twine &name) const {
    llvm::Value *const at =
        b_.CreateInBoundsGEP(b_.getDoubleTy(), column, index);
    if (!lanes_.vector()) {
      return b_.CreateLoad(lanes_.ty, at, name);
    }
    return b_.CreateMaskedLoad(lanes_.ty, at, llvm::Align(alignof(double)),
                               mask, constant(1.0), name);
  }

  void store(llvm::Value *v, llvm::Value *column, llvm::Value *index,
             llvm::Value *mask) const {
    llvm::Value *const at =
        b_.CreateInBoundsGEP(b_.getDoubleTy(), column, index);
    if (!lanes_.vector()) {
      b_.CreateStore(v, at);
      return;
    }
    b_.CreateMaskedStore(v, at, llvm::Align(alignof(double)), mask);
  }

  llvm::Value *unary(rt::OpCode op, llvm::Value *u) const {
    if (rt::lowering_of(op).how == rt::Lower::Call) {
      return call(op, {u});
    }
    if (op == rt::OpCode::Neg) {
      return b_.CreateFNeg(u);
    }
    // Sign: u > 0 ? 1 : u < 0 ? -1 : u - u -- the last arm reaches only ±0
    // and NaN.
    llvm::Value *const zero = constant(0.0);
    return b_.CreateSelect(b_.CreateFCmpOGT(u, zero), constant(1.0),
                           b_.CreateSelect(b_.CreateFCmpOLT(u, zero),
                                           constant(-1.0),
                                           b_.CreateFSub(u, u)));
  }

  // Not llvm.fmuladd, which is only *permission* to fuse
  llvm::Value *fma(const rt::Contraction &c, llvm::Value *x, llvm::Value *y,
                   llvm::Value *z) const {
    return b_.CreateIntrinsic(llvm::Intrinsic::fma, {lanes_.ty},
                              {c.negated ? b_.CreateFNeg(x) : x, y, z});
  }

  llvm::Value *binary(rt::OpCode op, llvm::Value *l, llvm::Value *r) const {
    switch (rt::lowering_of(op).how) {
    case rt::Lower::Infix:
      return op == rt::OpCode::Add   ? b_.CreateFAdd(l, r)
             : op == rt::OpCode::Mul ? b_.CreateFMul(l, r)
                                     : b_.CreateFDiv(l, r);
    // 1.0 or 0.0 in the lane type, as compare_impl computes.  Ordered, so a
    // NaN operand compares false.
    case rt::Lower::Compare:
      return b_.CreateUIToFP(b_.CreateFCmp(op == rt::OpCode::Lt
                                               ? llvm::CmpInst::FCMP_OLT
                                               : llvm::CmpInst::FCMP_OLE,
                                           l, r),
                             lanes_.ty);
    case rt::Lower::Call:
      return call(op, {l, r});
    // Max and Min: the NaN-propagating pair the interpreter matches, where
    // maxnum/minnum are not.
    case rt::Lower::Special:
    case rt::Lower::None:
      return b_.CreateBinaryIntrinsic(op == rt::OpCode::Max
                                          ? llvm::Intrinsic::maximum
                                          : llvm::Intrinsic::minimum,
                                      l, r);
    }
    std::unreachable();
  }

  // Every component of one call, lane by lane: [component * width + lane].
  using Answers = std::vector<llvm::Value *>;

  // One call per lane, the function having a scalar entry point only, as a
  // libm call does; inactive lanes call it too, at the 1.0 a masked load
  // reads, and their answers are never stored.  Every component is read back
  // from the one call, so a value and its partials cost one call per lane.
  Answers extern_call(const rt::ExternRecord &e, std::uint32_t index,
                      std::span<llvm::Value *const> args) const {
    llvm::Function *const kernel = b_.GetInsertBlock()->getParent();
    llvm::IRBuilder<> entry(&kernel->getEntryBlock(),
                            kernel->getEntryBlock().getFirstInsertionPt());
    llvm::Type *const f64 = b_.getDoubleTy();
    llvm::Type *const i64 = b_.getInt64Ty();
    const auto components = e.size();
    llvm::Value *const in = entry.CreateAlloca(
        f64, llvm::ConstantInt::get(i64, e.arity), "extern.in");
    llvm::Value *const answer = entry.CreateAlloca(
        f64, llvm::ConstantInt::get(i64, components), "extern.out");
    llvm::Value *const fn = b_.CreateLoad(
        b_.getPtrTy(),
        b_.CreateConstInBoundsGEP1_64(b_.getPtrTy(), externs_, index),
        "extern.fn");
    llvm::Function *const trampoline = trampoline_decl(m_);

    Answers out(components * lanes_.width);
    per_lane(args, [&](unsigned lane, llvm::ArrayRef<llvm::Value *> scalars) {
      for (const auto [j, a] : scalars | std::views::enumerate) {
        b_.CreateStore(a, b_.CreateConstInBoundsGEP1_64(
                              f64, in, static_cast<std::uint64_t>(j)));
      }
      b_.CreateCall(trampoline,
                    {fn, in, llvm::ConstantInt::get(i64, e.arity), answer,
                     llvm::ConstantInt::get(i64, components)});
      for (const std::size_t c : std::views::iota(0uz, components)) {
        out[c * lanes_.width + lane] =
            b_.CreateLoad(f64, b_.CreateConstInBoundsGEP1_64(f64, answer, c));
      }
    });
    return out;
  }
  // One component of a call's answers, back in the lane type.
  llvm::Value *component(const Answers &answers, std::uint32_t c) const {
    return gathered(
        [&](unsigned lane) { return answers[c * lanes_.width + lane]; });
  }

  // Both arms are already computed, so this is one blend: no branch and no
  // lane divergence.
  llvm::Value *ternary(rt::OpCode op, llvm::Value *c, llvm::Value *t,
                       llvm::Value *f) const {
    // UNE, not ONE: C's `!=` is unordered, so a NaN condition is true, which
    // is what select_impl computes.  The ordered form sends the kernel down the
    // other arm than the sweep, and only a raw value as a condition reaches it
    // -- a comparison's output is never NaN.
    return op == rt::OpCode::Select
               ? b_.CreateSelect(b_.CreateFCmpUNE(c, constant(0.0)), t, f)
               : llvm::PoisonValue::get(lanes_.ty);
  }

private:
  // An intrinsic is declared on the lane type and left to the backend to
  // unroll; a libm function has only a scalar entry point, so it is unrolled
  // here.  Declared once per opcode: getOrInsertDeclaration re-mangles the name
  // into a heap string per ask, and the emitter asks per node.
  llvm::Value *call(rt::OpCode op, llvm::ArrayRef<llvm::Value *> args) const {
    auto &[fn, scalarise] = callees_[std::to_underlying(op)];
    if (fn == nullptr) {
      const llvm::Intrinsic::ID id = intrinsic_for(op);
      if (id != llvm::Intrinsic::not_intrinsic) {
        fn = llvm::Intrinsic::getOrInsertDeclaration(&m_, id, {lanes_.ty});
      } else {
        fn =
            libm_decl(m_, rt::label_of(op), static_cast<unsigned>(args.size()));
        scalarise = lanes_.vector();
      }
    }
    return scalarise ? scalarised(fn, args) : libm_call(fn, args);
  }
  // A libm call stays that call: nobuiltin keeps the simplifier from trading
  // it for another function that rounds differently.
  llvm::Value *libm_call(llvm::Function *fn,
                         llvm::ArrayRef<llvm::Value *> args) const {
    llvm::CallInst *const c = b_.CreateCall(fn, args);
    if (!fn->isIntrinsic()) {
      c->addFnAttr(llvm::Attribute::NoBuiltin);
    }
    return c;
  }

  llvm::Value *scalarised(llvm::Function *fn,
                          llvm::ArrayRef<llvm::Value *> args) const {
    std::vector<llvm::Value *> answers(lanes_.width);
    per_lane(std::span{args.data(), args.size()},
             [&](unsigned lane, llvm::ArrayRef<llvm::Value *> scalars) {
               answers[lane] = libm_call(fn, scalars);
             });
    return gathered([&](unsigned lane) { return answers[lane]; });
  }

  // Lane by lane, each argument as a scalar: what a function with a scalar
  // entry point only -- libm's, or the program's -- is handed.
  void per_lane(std::span<llvm::Value *const> args, auto &&scalar) const {
    for (const unsigned lane : std::views::iota(0u, lanes_.width)) {
      llvm::SmallVector<llvm::Value *, 4> scalars;
      for (llvm::Value *const a : args) {
        scalars.push_back(lanes_.vector() ? b_.CreateExtractElement(a, lane)
                                          : a);
      }
      scalar(lane, scalars);
    }
  }
  // Each lane's scalar, back in the lane type.
  llvm::Value *gathered(auto &&of_lane) const {
    if (!lanes_.vector()) {
      return of_lane(0u);
    }
    llvm::Value *out = llvm::PoisonValue::get(lanes_.ty);
    for (const unsigned lane : std::views::iota(0u, lanes_.width)) {
      out = b_.CreateInsertElement(out, of_lane(lane), lane);
    }
    return out;
  }

  llvm::Module &m_;
  llvm::IRBuilder<> &b_;
  LaneType lanes_;
  llvm::Value *externs_;
  struct Callee {
    llvm::Function *fn = nullptr;
    bool scalarise = false;
  };
  mutable std::array<Callee, rt::op_count> callees_{};
};

llvm::Function *declare_kernel(llvm::Module &m, llvm::StringRef name,
                               const rt::Graph<double> &g) {
  llvm::LLVMContext &ctx = m.getContext();
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::PointerType *const ptr = llvm::PointerType::getUnqual(ctx);
  // xs, f, g, h, n, externs -- four column arrays, the batch length, and the
  // table of functions the program supplied.
  auto *const fty = llvm::FunctionType::get(
      llvm::Type::getVoidTy(ctx), {ptr, ptr, ptr, ptr, i64, ptr}, false);
  auto *const fn =
      llvm::Function::Create(fty, llvm::Function::ExternalLinkage, name, m);

  static constexpr std::array names{"xs", "f", "g", "h", "n", "externs"};
  for (const auto [i, arg_name] : names | std::views::enumerate) {
    fn->getArg(static_cast<unsigned>(i))->setName(arg_name);
  }

  // Everything here is about the four pointer *arrays*, not the columns they
  // hold: those are reached through a load, so no parameter attribute reaches
  // them and the kernel is not argmemonly.  All four are readonly -- the writes
  // go through loaded pointers, never back into the arrays.  A block that was
  // not asked for arrives as an empty span, whose data() may be null, so it
  // gets neither nonnull nor a size.
  const auto &layout = g.layout();
  const std::array counts{g.arity(),      layout.values,  layout.jacobian,
                          layout.hessian, std::size_t{0}, g.externs().size()};
  for (const auto [i, count] : counts | std::views::enumerate) {
    const auto arg = static_cast<unsigned>(i);
    if (arg == 4) {
      continue; // n: a value, below
    }
    fn->addParamAttr(arg, llvm::Attribute::NoAlias);
    fn->addParamAttr(arg, llvm::Attribute::NoCapture);
    fn->addParamAttr(arg, llvm::Attribute::ReadOnly);
    fn->addParamAttr(arg, llvm::Attribute::NoUndef);
    fn->addParamAttr(arg, llvm::Attribute::getWithAlignment(
                              ctx, llvm::Align(alignof(void *))));
    if (count != 0) {
      fn->addParamAttr(arg, llvm::Attribute::NonNull);
      fn->addDereferenceableParamAttr(arg, count * sizeof(void *));
    }
  }
  fn->addParamAttr(4, llvm::Attribute::NoUndef);

  // nounwind, or the optimiser cannot move code across the libm calls.
  fn->setDoesNotThrow();
  fn->setWillReturn();
  fn->setMustProgress();
  return fn;
}

// Loaded once in the entry block, so they are loop-invariant.
using ColumnValues = boost::container::small_vector<llvm::Value *, 8>;
struct HoistedColumns {
  ColumnValues inputs;
  // f[k], g[k*n + j], h[c*n + i]
  rt::PerBlock<ColumnValues> out;
};

HoistedColumns hoist_columns(llvm::IRBuilder<> &b, llvm::Function &fn,
                             const rt::Graph<double> &g) {
  llvm::PointerType *const ptr = llvm::PointerType::getUnqual(fn.getContext());

  const auto load_columns = [&](unsigned arg, std::size_t count,
                                const char *stem) {
    return std::views::iota(0uz, count) |
           std::views::transform([&](std::size_t j) {
             llvm::Value *const slot =
                 b.CreateConstInBoundsGEP1_64(ptr, fn.getArg(arg), j);
             return b.CreateLoad(ptr, slot, stem + llvm::Twine(j));
           }) |
           impl::to<ColumnValues>();
  };

  const auto &layout = g.layout();
  return {.inputs = load_columns(0, g.arity(), "col"),
          .out = {.values = load_columns(1, layout.values, "f"),
                  .jacobian = load_columns(2, layout.jacobian, "g"),
                  .hessian = load_columns(3, layout.hessian, "h")}};
}

// Ids are topological, so one pass needs no worklist.  Under contraction the
// adds carry their product and the multiplies they swallowed are never reached.
[[nodiscard]] std::vector<llvm::Value *>
emit_nodes(const Emitter &emit, const rt::Graph<double> &g,
           const HoistedColumns &cols, llvm::Value *index, llvm::Value *mask) {
  std::vector<llvm::Value *> value(g.size(), nullptr);
  std::map<std::pair<std::uint32_t, rt::NodeId>, Emitter::Answers> called;
  for (const rt::Step &s : g.schedule()) {
    const auto &[v, c] = std::pair{s.node, s.fma};
    const auto &p = g[v];
    if (c) {
      value[v] = emit.fma(c, value[c.x], value[c.y], value[c.z]);
      continue;
    }
    const auto operands = g.operands(v);
    switch (rt::kind_of(p.op)) {
    case rt::Kind::Const:
      value[v] = emit.constant(p.value);
      break;
    case rt::Kind::Input:
      value[v] = emit.load(cols.inputs[rt::input_column(
                               g.inputs(), p.op, rt::detail::index_of(p))],
                           index, mask, "");
      break;
    case rt::Kind::Nary: {
      // The terms into the running sum in list order, a product term as the
      // fma the sweep takes where the step is fused; the first term is formed.
      llvm::Value *acc = nullptr;
      for (const rt::NodeId t : rt::arguments(g, v)) {
        const rt::Contraction c =
            s.fused ? rt::product_of(g, t) : rt::Contraction{};
        if (!c) {
          acc = acc ? emit.binary(rt::OpCode::Add, acc, value[t]) : value[t];
        } else if (acc) {
          acc = emit.fma(c, value[c.x], value[c.y], acc);
        } else {
          acc = emit.binary(rt::OpCode::Mul, value[c.x], value[c.y]);
          acc = c.negated ? emit.unary(rt::OpCode::Neg, acc) : acc;
        }
      }
      value[v] = acc;
      break;
    }
    case rt::Kind::Extern: {
      // One call per (function, argument list), whichever component asked
      // first; the rest are loads off the same answer.
      const auto [fn, component] = rt::detail::extern_of(p);
      auto [answers, fresh] = called.try_emplace({fn, operands[0]});
      if (fresh) {
        const auto args =
            rt::arguments(g, v) |
            std::views::transform([&value](rt::NodeId a) { return value[a]; }) |
            impl::to<std::vector<llvm::Value *>>();
        answers->second = emit.extern_call(g.externs()[fn], fn, args);
      }
      value[v] = emit.component(answers->second, component);
      break;
    }
    case rt::Kind::Unary:
      value[v] = emit.unary(p.op, value[operands[0]]);
      break;
    case rt::Kind::Ternary:
      value[v] = emit.ternary(p.op, value[operands[0]], value[operands[1]],
                              value[operands[2]]);
      break;
    case rt::Kind::Binary:
      value[v] = emit.binary(p.op, value[operands[0]], value[operands[1]]);
      break;
    case rt::Kind::Tensor:
      std::unreachable(); // Compiler::compile refuses a graph with tensor rows
    }
  }
  return value;
}

void emit_stores(const Emitter &emit, const rt::Graph<double> &g,
                 const HoistedColumns &cols,
                 std::span<llvm::Value *const> value, llvm::Value *index,
                 llvm::Value *mask) {
  const rt::PerBlock<std::span<llvm::Value *const>> columns{
      cols.out.values, cols.out.jacobian, cols.out.hessian};
  for (const auto [block_columns, block] :
       rt::zip_blocks(columns, g.output_blocks())) {
    for (const auto [column, o] : std::views::zip(block_columns, block)) {
      emit.store(value[o], column, index, mask);
    }
  }
}

} // namespace

std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<double> &g,
            llvm::StringRef name, const unsigned width,
            const llvm::DataLayout &layout, llvm::StringRef triple) {
  auto m = std::make_unique<llvm::Module>("ddx.jit", ctx);
  // Before the first instruction: IRBuilder reads the layout for every
  // alignment it fills in without being told one.
  m->setDataLayout(layout);
  m->setTargetTriple(triple);
  llvm::Function *const fn = declare_kernel(*m, name, g);
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::Type *const f64 = llvm::Type::getDoubleTy(ctx);
  llvm::Argument *const count = fn->getArg(4);
  const LaneType lanes{.width = width,
                       .ty = width > 1 ? llvm::FixedVectorType::get(f64, width)
                                       : f64};

  auto *const entry = llvm::BasicBlock::Create(ctx, "entry", fn);
  auto *const loop = llvm::BasicBlock::Create(ctx, "loop", fn);
  auto *const exit = llvm::BasicBlock::Create(ctx, "exit", fn);

  // No fast-math flags at all: contraction is decided in the graph and spelled
  // llvm.fma, so allowContract would fuse a *second* set the sweep computed
  // separately.
  llvm::IRBuilder<> b(entry);
  const HoistedColumns cols = hoist_columns(b, *fn, g);

  // n == 0 is a degenerate call, and saying so is what keeps the loop out of
  // the cold half of the layout.
  llvm::MDBuilder md(ctx);
  b.CreateCondBr(b.CreateICmpEQ(count, llvm::ConstantInt::get(i64, 0)), exit,
                 loop, md.createBranchWeights(1, 1U << 12));

  b.SetInsertPoint(loop);
  llvm::PHINode *const index = b.CreatePHI(i64, 2, "i");
  index->addIncoming(llvm::ConstantInt::get(i64, 0), entry);

  // Lane k is live while k < n - i.  Signed on purpose: both sides are
  // non-negative, and it is one instruction where the saturating
  // llvm.get.active.lane.mask is a dozen on AVX2.
  llvm::Value *mask = nullptr;
  if (lanes.vector()) {
    // nuw: i < n on every entry to the body, so the difference never wraps.
    llvm::Value *const remaining =
        b.CreateSub(count, index, "remaining", /*HasNUW=*/true);
    llvm::Value *const step =
        b.CreateStepVector(llvm::FixedVectorType::get(i64, width));
    mask = b.CreateICmpSLT(step, b.CreateVectorSplat(width, remaining), "mask");
  }

  const Emitter emit(*m, b, lanes, fn->getArg(5));
  const std::vector<llvm::Value *> value =
      emit_nodes(emit, g, cols, index, mask);
  emit_stores(emit, g, cols, value, index, mask);

  // nuw/nsw: n counts doubles the caller has allocated, so n + W is nowhere
  // near either bound.  It is what lets SCEV name an exact trip count.
  llvm::Value *const next =
      b.CreateAdd(index, llvm::ConstantInt::get(i64, width), "i.next",
                  /*HasNUW=*/true, /*HasNSW=*/true);
  index->addIncoming(next, loop);
  b.CreateCondBr(b.CreateICmpUGE(next, count), exit, loop);

  b.SetInsertPoint(exit);
  b.CreateRetVoid();

  return llvm::verifyFunction(*fn, &llvm::errs()) ? nullptr : std::move(m);
}

} // namespace ddx::jit::detail
