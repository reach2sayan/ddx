#include "codegen.hpp"

#include "spelling.hpp"
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
#include <concepts>
#include <cstdint>
#include <map>
#include <optional>
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
      const auto &[form, name] = rt::spelling_of(static_cast<rt::OpCode>(i));
      llvm::SmallString<32> spelled{"llvm."};
      spelled += name;
      // Not pow: the library-call simplifier rewrites llvm.pow with a constant
      // operand -- 1/x for -1, sqrt for 0.5 -- and libm's pow is not
      // correctly rounded, so the kernel would leave the sweep by an ulp.
      t[i] = form == rt::Form::Call && i != std::to_underlying(rt::OpCode::Pow)
                 ? llvm::Intrinsic::lookupIntrinsicID(spelled)
                 : llvm::Intrinsic::not_intrinsic;
    }
    return t;
  }();
  return table[std::to_underlying(op)];
}

// A function the kernel calls out to, which neither unwinds nor loops;
// `effects`, where known, is what lets the optimiser move the call.
llvm::Function *declare(llvm::Module &m, llvm::StringRef name,
                        llvm::FunctionType *type,
                        std::optional<llvm::MemoryEffects> effects = {}) {
  auto *fn =
      llvm::cast<llvm::Function>(m.getOrInsertFunction(name, type).getCallee());
  if (effects) {
    fn->setMemoryEffects(*effects);
  }
  fn->setDoesNotThrow();
  fn->setWillReturn();
  return fn;
}

// The IR spelling of -fno-math-errno: without it a call is assumed to write
// errno, which blocks hoisting across it.
llvm::Function *libm_decl(llvm::Module &m, std::string_view name,
                          llvm::Type *scalar, unsigned args) {
  llvm::SmallVector<llvm::Type *, 2> params(args, scalar);
  auto *fn = declare(m, name, llvm::FunctionType::get(scalar, params, false),
                     llvm::MemoryEffects::none());
  fn->setDoesNotFreeMemory();
  return fn;
}

// A way out of a kernel into the runtime, defined beside libm: the extern
// trampoline, handed the function's address as a constant, and the
// tensor step and store.
template <CKernelScalar T>
llvm::Function *runtime_decl(llvm::Module &m, std::string_view symbol,
                             llvm::ArrayRef<llvm::Value *> args) {
  const auto types = llvm::to_vector(
      llvm::map_range(args, [](llvm::Value *a) { return a->getType(); }));
  return declare(m, libm_name<T>(symbol),
                 llvm::FunctionType::get(llvm::Type::getVoidTy(m.getContext()),
                                         types, false));
}

// W points per iteration, as a T at W == 1 and <W x T> otherwise.  Every
// IRBuilder operation is typed on `ty`, so only the memory accesses and the
// calls with no vector intrinsic tell the two apart.
struct LaneType {
  unsigned width;
  llvm::Type *scalar;
  llvm::Type *ty;
  [[nodiscard]] constexpr bool vector() const noexcept { return width > 1; }
};

// The kernel's arguments a tensor step is handed: the TensorCall, and the
// interpreter's two tapes.  Null in a scalar graph.
struct TensorTapes {
  llvm::Value *call = nullptr;
  llvm::Value *tape = nullptr;
  llvm::Value *vtape = nullptr;
};

template <CKernelScalar T> class Emitter {
public:
  Emitter(llvm::Module &m, llvm::IRBuilder<> &b, LaneType lanes,
          TensorTapes tensors)
      : m_{m}, b_{b}, lanes_{lanes}, tensors_{tensors} {}

  // A splat where the lane type is a vector.
  [[nodiscard]] llvm::Value *constant(double v) const {
    return llvm::ConstantFP::get(lanes_.ty, v);
  }

  // Columns are plain T*, so no alignment past the element's is claimed.
  // A load past the batch reads inactive lanes as 1.0: never stored, and inside
  // every op's domain, so a scalarised libm call cannot hit a pole.
  [[nodiscard]] llvm::Value *load(llvm::Value *column, llvm::Value *index,
                                  llvm::Value *mask,
                                  const llvm::Twine &name) const {
    llvm::Value *const at = b_.CreateInBoundsGEP(lanes_.scalar, column, index);
    if (!lanes_.vector()) {
      return b_.CreateLoad(lanes_.ty, at, name);
    }
    return b_.CreateMaskedLoad(lanes_.ty, at, llvm::Align(alignof(T)), mask,
                               constant(1.0), name);
  }

  void store(llvm::Value *v, llvm::Value *column, llvm::Value *index,
             llvm::Value *mask) const {
    llvm::Value *const at = b_.CreateInBoundsGEP(lanes_.scalar, column, index);
    if (!lanes_.vector()) {
      b_.CreateStore(v, at);
      return;
    }
    b_.CreateMaskedStore(v, at, llvm::Align(alignof(T)), mask);
  }

  llvm::Value *unary(rt::OpCode op, llvm::Value *u) const {
    if (rt::spelling_of(op).form == rt::Form::Call) {
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
    switch (rt::spelling_of(op).form) {
    case rt::Form::Infix:
      return op == rt::OpCode::Add   ? b_.CreateFAdd(l, r)
             : op == rt::OpCode::Mul ? b_.CreateFMul(l, r)
                                     : b_.CreateFDiv(l, r);
    // 1.0 or 0.0 in the lane type, as compare_impl computes.  Ordered, so a
    // NaN operand compares false.
    case rt::Form::Compare:
      return b_.CreateUIToFP(b_.CreateFCmp(op == rt::OpCode::Lt
                                               ? llvm::CmpInst::FCMP_OLT
                                               : llvm::CmpInst::FCMP_OLE,
                                           l, r),
                             lanes_.ty);
    case rt::Form::Call:
      return call(op, {l, r});
    // Max and Min: the NaN-propagating pair the interpreter matches, where
    // maxnum/minnum are not.
    case rt::Form::Special:
    case rt::Form::None:
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
  Answers extern_call(const rt::ExternEntry<T> &e,
                      std::span<llvm::Value *const> args) const {
    llvm::Function *const kernel = b_.GetInsertBlock()->getParent();
    llvm::IRBuilder<> entry(&kernel->getEntryBlock(),
                            kernel->getEntryBlock().getFirstInsertionPt());
    llvm::Type *const scalar = lanes_.scalar;
    llvm::Type *const i64 = b_.getInt64Ty();
    const auto components = e.size();
    llvm::Value *const in = entry.CreateAlloca(
        scalar, llvm::ConstantInt::get(i64, e.arity), "extern.in");
    llvm::Value *const answer = entry.CreateAlloca(
        scalar, llvm::ConstantInt::get(i64, components), "extern.out");
    llvm::Value *const fn = llvm::ConstantExpr::getIntToPtr(
        llvm::ConstantInt::get(i64,
                               reinterpret_cast<std::uintptr_t>(e.fn.get())),
        b_.getPtrTy());

    Answers out(components * lanes_.width);
    per_lane(args, [&](unsigned lane, llvm::ArrayRef<llvm::Value *> scalars) {
      for (const auto [j, a] : scalars | std::views::enumerate) {
        b_.CreateStore(a, b_.CreateConstInBoundsGEP1_64(
                              scalar, in, static_cast<std::uint64_t>(j)));
      }
      runtime_call(trampoline_symbol,
                   {fn, in, llvm::ConstantInt::get(i64, e.arity), answer,
                    llvm::ConstantInt::get(i64, components)});
      for (const std::size_t c : std::views::iota(0uz, components)) {
        out[c * lanes_.width + lane] = b_.CreateLoad(
            scalar, b_.CreateConstInBoundsGEP1_64(scalar, answer, c));
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
  llvm::Value *ternary(llvm::Value *c, llvm::Value *t, llvm::Value *f) const {
    // UNE, not ONE: C's `!=` is unordered, so a NaN condition is true, which
    // is what select_impl computes.  The ordered form sends the kernel down the
    // other arm than the sweep, and only a raw value as a condition reaches it
    // -- a comparison's output is never NaN.
    return b_.CreateSelect(b_.CreateFCmpUNE(c, constant(0.0)), t, f);
  }

  // Lane k of tape row r at tape + r*W + k, as the interpreter lays it out.
  void to_tape(llvm::Value *v, std::uint32_t row, llvm::Value *mask) const {
    store(v, tensors_.tape, lanes_at(row), mask);
  }
  [[nodiscard]] llvm::Value *from_tape(std::uint32_t row,
                                       llvm::Value *mask) const {
    return load(tensors_.tape, lanes_at(row), mask, "tensor.out");
  }

  void tensor_step(rt::NodeId v) const {
    runtime_call(tensor_step_symbol,
                 {tensors_.call, b_.getInt64(v), b_.getInt64(lanes_.width),
                  tensors_.tape, tensors_.vtape});
  }
  void tensor_store(llvm::Function &kernel, llvm::Value *index,
                    llvm::Value *live) const {
    runtime_call(tensor_store_symbol,
                 {tensors_.call, kernel.getArg(1), kernel.getArg(2),
                  kernel.getArg(3), index, live, b_.getInt64(lanes_.width),
                  tensors_.tape, tensors_.vtape});
  }

private:
  void runtime_call(std::string_view symbol,
                    llvm::ArrayRef<llvm::Value *> args) const {
    b_.CreateCall(runtime_decl<T>(m_, symbol, args), args);
  }

  [[nodiscard]] llvm::Value *lanes_at(std::uint32_t row) const {
    return b_.getInt64(std::uint64_t{row} * lanes_.width);
  }

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
        fn = libm_decl(m_, libm_name<T>(rt::label_of(op)), lanes_.scalar,
                       static_cast<unsigned>(args.size()));
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
  TensorTapes tensors_;
  struct Callee {
    llvm::Function *fn = nullptr;
    bool scalarise = false;
  };
  mutable std::array<Callee, rt::op_count> callees_{};
};

// Tensor nodes as loops: one nest per row the kernel keeps, over its columns
// and then its contiguous rows, the inner loop marked for the loop vectoriser
// -- which, forced, may also reorder a reduction's adds.  A node the graph
// fused into its reader is its element expression inside the reader's loop.
// MatMul and StackRows stay the interpreter's step.  Lane k of a W-lane kernel
// is one pass of a loop over k.
template <CKernelScalar T> class TensorLoops {
public:
  TensorLoops(llvm::Module &m, llvm::IRBuilder<> &b, const rt::Graph<T> &g,
              const Emitter<T> &elem, LaneType lanes, TensorTapes tapes)
      : m_{m}, b_{b}, g_{g}, elem_{elem}, lanes_{lanes}, tapes_{tapes},
        tables_(g.size(), nullptr), stepped_(g.size(), false),
        done_(g.size(), false) {
    // In the entry block, once per call.
    auto *const decl =
        declare(m, libm_name<T>(tensor_table_symbol),
                llvm::FunctionType::get(b.getPtrTy(),
                                        {b.getPtrTy(), b.getInt64Ty()}, false),
                llvm::MemoryEffects::readOnly());
    for (const rt::NodeId v :
         g.schedule() | std::views::transform(&rt::Step::node)) {
      if (g.op_of(v) == rt::OpCode::VecConst ||
          g.op_of(v) == rt::OpCode::MatConst) {
        tables_[v] = b.CreateCall(
            decl,
            {tapes.call,
             b.getInt64(rt::detail::get<rt::TensorRef>(g[v]).value)},
            "table");
      }
      // A step and the output scatter read rows; a loop reads a table in place.
      if (!handles(v)) {
        rt::detail::for_each_operand(g, v,
                                     [&](rt::NodeId u) { stepped_[u] = true; });
      }
    }
    for (const rt::NodeId o : g.outputs()) {
      stepped_[o] = true;
    }
  }

  // Whether node v is a loop here rather than the interpreter's step.
  [[nodiscard]] bool handles(rt::NodeId v) const {
    return g_.op_of(v) != rt::OpCode::MatMul &&
           g_.op_of(v) != rt::OpCode::StackRows;
  }

  // Node v at every lane; a scalar it yields is left in its tape row.
  void emit(rt::NodeId v, std::span<llvm::Value *const> value) {
    value_ = value;
    if (lanes_.width == 1) {
      k_ = i64(0);
      lane(v);
      return;
    }
    loop(lanes_.width, {}, false, [&](llvm::Value *k, auto) {
      k_ = k;
      lane(v);
      return Carried{};
    });
  }

private:
  using Carried = llvm::SmallVector<llvm::Value *, 2>;

  [[nodiscard]] llvm::Value *i64(std::uint64_t c) const {
    return b_.getInt64(c);
  }
  [[nodiscard]] rt::NodeId operand(rt::NodeId v, std::size_t slot) const {
    return g_.operands(v)[slot];
  }
  [[nodiscard]] std::uint64_t rows(rt::NodeId u) const {
    return g_.shape(u).rows;
  }

  // for (iv = 0; iv < n; ++iv) with `init` carried through `body`, which
  // answers the next values; the finals come back.  n is a constant, the
  // shapes being the graph's.
  Carried loop(std::uint64_t n, llvm::ArrayRef<llvm::Value *> init,
               bool vectorise, auto &&body) {
    if (n == 0) {
      return {init.begin(), init.end()};
    }
    llvm::LLVMContext &ctx = m_.getContext();
    llvm::BasicBlock *const pre = b_.GetInsertBlock();
    llvm::Function *const fn = pre->getParent();
    auto *const head = llvm::BasicBlock::Create(ctx, "t.loop", fn);
    auto *const done = llvm::BasicBlock::Create(ctx, "t.done", fn);
    b_.CreateBr(head);
    b_.SetInsertPoint(head);
    llvm::PHINode *const iv = b_.CreatePHI(b_.getInt64Ty(), 2, "t.i");
    iv->addIncoming(i64(0), pre);
    llvm::SmallVector<llvm::PHINode *, 2> phis;
    Carried carried;
    for (llvm::Value *const v : init) {
      phis.push_back(b_.CreatePHI(v->getType(), 2));
      phis.back()->addIncoming(v, pre);
      carried.push_back(phis.back());
    }
    const Carried next = body(static_cast<llvm::Value *>(iv), carried);
    llvm::Value *const inc = b_.CreateAdd(iv, i64(1), "", true, true);
    llvm::BasicBlock *const latch = b_.GetInsertBlock();
    iv->addIncoming(inc, latch);
    for (const auto [phi, v] : std::views::zip(phis, next)) {
      phi->addIncoming(v, latch);
    }
    llvm::BranchInst *const br =
        b_.CreateCondBr(b_.CreateICmpULT(inc, i64(n)), head, done);
    if (vectorise) {
      llvm::Metadata *const enable[] = {
          llvm::MDString::get(ctx, "llvm.loop.vectorize.enable"),
          llvm::ConstantAsMetadata::get(b_.getTrue())};
      // Unrolled whole first, a reduction is a serial chain the vectoriser
      // never sees.
      llvm::Metadata *const keep[] = {
          llvm::MDString::get(ctx, "llvm.loop.unroll.disable")};
      llvm::Metadata *const ops[] = {nullptr, llvm::MDNode::get(ctx, enable),
                                     llvm::MDNode::get(ctx, keep)};
      llvm::MDNode *const id = llvm::MDNode::getDistinct(ctx, ops);
      id->replaceOperandWith(0, id);
      br->setMetadata(llvm::LLVMContext::MD_loop, id);
    }
    b_.SetInsertPoint(done);
    return next;
  }
  // Every (i, j) of an r x c shape, i innermost.
  Carried each(rt::Shape s, llvm::ArrayRef<llvm::Value *> init, auto &&body) {
    return loop(s.cols, init, false, [&](llvm::Value *j, const Carried &outer) {
      return loop(s.rows, outer, true, [&](llvm::Value *i, const Carried &c) {
        return body(i, j, c);
      });
    });
  }

  [[nodiscard]] llvm::Value *add(llvm::Value *a, llvm::Value *b) const {
    return b_.CreateAdd(a, b);
  }
  [[nodiscard]] llvm::Value *flat(llvm::Value *i, llvm::Value *j,
                                  std::uint64_t r) const {
    return add(i, b_.CreateMul(j, i64(r)));
  }
  [[nodiscard]] llvm::Value *element(llvm::Value *base, llvm::Value *at) const {
    return b_.CreateInBoundsGEP(lanes_.scalar, base, at);
  }
  [[nodiscard]] llvm::Value *read(llvm::Value *base, llvm::Value *at) const {
    return b_.CreateLoad(lanes_.scalar, element(base, at));
  }
  // Lane k of u's row.
  [[nodiscard]] llvm::Value *row(rt::NodeId u) const {
    const rt::TensorRow &r = g_.tensors().rows[u];
    return element(tapes_.vtape, add(i64(std::uint64_t{r.at} * lanes_.width),
                                     b_.CreateMul(k_, i64(r.width))));
  }
  // Lane k of a scalar node, which is SSA: its tape row may be another
  // node's by the time a fused reader runs.
  [[nodiscard]] llvm::Value *scalar(rt::NodeId u) const {
    return lanes_.width == 1 ? value_[u]
                             : b_.CreateExtractElement(value_[u], k_);
  }
  void scalar_out(rt::NodeId v, llvm::Value *x) const {
    b_.CreateStore(
        x, element(tapes_.tape,
                   add(i64(std::uint64_t{g_.rows()[v]} * lanes_.width), k_)));
  }

  // Element (i, j) of u: off a table, off u's row, or computed here.
  llvm::Value *at(rt::NodeId u, llvm::Value *i, llvm::Value *j) {
    switch (g_.op_of(u)) {
    case rt::OpCode::VecConst:
      return read(tables_[u], i);
    case rt::OpCode::MatRow:
      if (const rt::NodeId a = operand(u, 0);
          g_.op_of(a) == rt::OpCode::MatConst) {
        const bool plain =
            rt::detail::get<rt::Orientation>(g_[u]) == rt::Orientation::plain;
        return read(tables_[a],
                    plain ? flat(i, j, rows(a)) : flat(j, i, rows(a)));
      }
      break;
    default:
      break;
    }
    if (g_.fused_into(u) != rt::no_node) {
      return expr(u, i, j);
    }
    return read(row(u), flat(i, j, rows(u)));
  }
  llvm::Value *at(rt::NodeId u, llvm::Value *i) { return at(u, i, i64(0)); }

  // What an elementwise node is at (i, j), in the operand order the
  // interpreter multiplies in.
  llvm::Value *expr(rt::NodeId v, llvm::Value *i, llvm::Value *j) {
    const auto [a, b, c] = g_.operands(v);
    switch (g_.op_of(v)) {
    case rt::OpCode::Map:
      return elem_.unary(rt::detail::get<rt::OpCode>(g_[v]), at(a, i, j));
    case rt::OpCode::Zip:
      return elem_.binary(rt::detail::get<rt::OpCode>(g_[v]), at(a, i, j),
                          at(b, i, j));
    case rt::OpCode::Scale:
      return b_.CreateFMul(scalar(a), at(b, i, j));
    case rt::OpCode::Outer:
      return b_.CreateFMul(at(b, j), at(a, i));
    case rt::OpCode::RowScale:
      return b_.CreateFMul(at(a, i), at(b, i, j));
    case rt::OpCode::ColScale:
      return b_.CreateFMul(at(b, j), at(a, i, j));
    case rt::OpCode::VSelect:
      return elem_.ternary(rt::detail::is_tensor(g_[a]) ? at(a, i, j)
                                                        : scalar(a),
                           at(b, i, j), at(c, i, j));
    case rt::OpCode::Diag:
      return b_.CreateSelect(b_.CreateICmpEQ(i, j), at(a, i),
                             elem_.constant(0.0));
    case rt::OpCode::MatRow: // a transpose, of a table or a computed matrix
      return g_.op_of(a) == rt::OpCode::MatConst ? at(v, i, j) : at(a, j, i);
    default:
      std::unreachable();
    }
  }

  llvm::Value *fmuladd(llvm::Value *x, llvm::Value *y, llvm::Value *z) const {
    return b_.CreateIntrinsic(llvm::Intrinsic::fmuladd, {lanes_.scalar},
                              {x, y, z});
  }

  // M x into `plain`'s row and M^T x into `transposed`'s, either absent,
  // four columns a pass: each element of M is read once for both, out is
  // loaded and stored a quarter as often, and x[i] once for four dots.
  void products(rt::NodeId m, rt::NodeId x, rt::NodeId plain,
                rt::NodeId transposed) {
    const rt::Shape s = g_.shape(m);
    llvm::Value *const py = plain != rt::no_node ? row(plain) : nullptr;
    llvm::Value *const ty =
        transposed != rt::no_node ? row(transposed) : nullptr;
    if (py != nullptr) {
      b_.CreateMemSet(py, b_.getInt8(0), s.rows * sizeof(T),
                      llvm::Align(alignof(T)));
    }
    const auto columns = [&](llvm::Value *j0, std::uint64_t count) {
      llvm::SmallVector<llvm::Value *, 4> js, xs;
      for (std::uint64_t d = 0; d < count; ++d) {
        js.push_back(add(j0, i64(d)));
        xs.push_back(py != nullptr ? at(x, js.back()) : nullptr);
      }
      const Carried zeros(ty != nullptr ? count : 0, elem_.constant(0.0));
      const Carried dots =
          loop(s.rows, zeros, true, [&](llvm::Value *i, const Carried &acc) {
            llvm::Value *const xi = ty != nullptr ? at(x, i) : nullptr;
            llvm::Value *const to = py != nullptr ? element(py, i) : nullptr;
            llvm::Value *y =
                to != nullptr ? b_.CreateLoad(lanes_.scalar, to) : nullptr;
            Carried next;
            for (std::uint64_t d = 0; d < count; ++d) {
              llvm::Value *const mij = at(m, i, js[d]);
              if (to != nullptr) {
                y = fmuladd(mij, xs[d], y);
              }
              if (xi != nullptr) {
                next.push_back(fmuladd(mij, xi, acc[d]));
              }
            }
            if (to != nullptr) {
              b_.CreateStore(y, to);
            }
            return next;
          });
      for (const auto [j, dot] : std::views::zip(js, dots)) {
        b_.CreateStore(dot, element(ty, j));
      }
    };
    loop(s.cols / 4, {}, false, [&](llvm::Value *q, auto) {
      columns(b_.CreateMul(q, i64(4)), 4);
      return Carried{};
    });
    if (s.cols % 4 != 0) {
      columns(i64(s.cols / 4 * 4), s.cols % 4);
    }
  }

  void lane(rt::NodeId v) {
    const auto [a, b, c] = g_.operands(v);
    const auto &p = g_[v];
    const rt::Shape s = g_.shape(v);
    llvm::Value *const zero = elem_.constant(0.0);
    switch (p.op) {
    case rt::OpCode::MatConst:
      return;
    case rt::OpCode::MatRow:
      // A table is read in place: copied only where it has a row -- an output
      // -- or, transposed, where a step reads that row.
      if (g_.op_of(a) == rt::OpCode::MatConst &&
          (g_.tensors().rows[v].width == 0 ||
           (rt::detail::get<rt::Orientation>(p) ==
                rt::Orientation::transposed &&
            !stepped_[v]))) {
        return;
      }
      break;
    case rt::OpCode::VecConst:
      if (!stepped_[v]) {
        return;
      }
      b_.CreateMemCpy(row(v), llvm::Align(alignof(T)), tables_[v],
                      llvm::Align(alignof(T)), s.size() * sizeof(T));
      return;
    case rt::OpCode::Pack:
      for (const auto [e, arg] : rt::arguments(g_, v) | std::views::enumerate) {
        b_.CreateStore(scalar(arg),
                       element(row(v), i64(static_cast<std::uint64_t>(e))));
      }
      return;
    case rt::OpCode::Gather:
      scalar_out(v, at(a, i64(rt::detail::get<rt::Element>(p).value)));
      return;
    case rt::OpCode::AddAt: {
      llvm::Value *const out = row(v);
      loop(s.size(), {}, true, [&](llvm::Value *i, auto) {
        b_.CreateStore(at(a, i), element(out, i));
        return Carried{};
      });
      llvm::Value *const to =
          element(out, i64(rt::detail::get<rt::Element>(p).value));
      b_.CreateStore(b_.CreateFAdd(b_.CreateLoad(lanes_.scalar, to), scalar(b)),
                     to);
      return;
    }
    case rt::OpCode::Sum:
    case rt::OpCode::Dot:
      scalar_out(
          v, each(g_.shape(a), {zero}, [&](auto i, auto j, const auto &acc) {
            return Carried{p.op == rt::OpCode::Sum
                               ? b_.CreateFAdd(acc[0], at(a, i, j))
                               : fmuladd(at(a, i, j), at(b, i, j), acc[0])};
          })[0]);
      return;
    case rt::OpCode::MatVec: {
      // A pair is one nest, at whichever of the two comes first.
      const rt::NodeId mate = g_.pairs()[v];
      if (done_[v]) {
        return;
      }
      const bool transposed =
          rt::detail::get<rt::Orientation>(p) == rt::Orientation::transposed;
      if (mate != rt::no_node) {
        done_[mate] = true;
      }
      products(a, b, transposed ? mate : v, transposed ? v : mate);
      return;
    }
    case rt::OpCode::RowOf: {
      llvm::Value *const out = row(v);
      llvm::Value *const r = i64(rt::detail::get<rt::Element>(p).value);
      loop(s.rows, {}, true, [&](llvm::Value *j, auto) {
        b_.CreateStore(at(a, r, j), element(out, j));
        return Carried{};
      });
      return;
    }
    default:
      break;
    }
    // Everything elementwise, as a row of its own.
    llvm::Value *const out = row(v);
    each(s, {}, [&](llvm::Value *i, llvm::Value *j, auto) {
      b_.CreateStore(expr(v, i, j), element(out, flat(i, j, s.rows)));
      return Carried{};
    });
  }

  llvm::Module &m_;
  llvm::IRBuilder<> &b_;
  const rt::Graph<T> &g_;
  const Emitter<T> &elem_;
  LaneType lanes_;
  TensorTapes tapes_;
  std::vector<llvm::Value *> tables_; // by id: a table's elements
  std::vector<bool> stepped_;         // by id: read as a row, not in place
  std::vector<bool> done_;            // by id: formed with its pair
  std::span<llvm::Value *const> value_;
  llvm::Value *k_ = nullptr;
};

template <CKernelScalar T>
llvm::Function *declare_kernel(llvm::Module &m, llvm::StringRef name,
                               const rt::Graph<T> &g) {
  llvm::LLVMContext &ctx = m.getContext();
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::PointerType *const ptr = llvm::PointerType::getUnqual(ctx);
  // xs, f, g, h, n, tensors, tapes -- four column arrays, the batch length,
  // and what the tensor steps read.
  auto *const fty = llvm::FunctionType::get(
      llvm::Type::getVoidTy(ctx), {ptr, ptr, ptr, ptr, i64, ptr, ptr}, false);
  auto *const fn =
      llvm::Function::Create(fty, llvm::Function::ExternalLinkage, name, m);

  static constexpr std::array names{"xs", "f",       "g",    "h",
                                    "n",  "tensors", "tapes"};
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
  const std::array counts{g.arity(), layout.values, layout.jacobian,
                          layout.hessian};
  for (const auto [i, count] : counts | std::views::enumerate) {
    const auto arg = static_cast<unsigned>(i);
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

template <CKernelScalar T>
HoistedColumns hoist_columns(llvm::IRBuilder<> &b, llvm::Function &fn,
                             const rt::Graph<T> &g) {
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
// A tensor node is a loop nest, or the interpreter's step over the scalars it
// reads stored to the tape; a scalar either yields is read back.
template <CKernelScalar T>
[[nodiscard]] std::vector<llvm::Value *>
emit_nodes(const Emitter<T> &emit, TensorLoops<T> *loops, const rt::Graph<T> &g,
           const HoistedColumns &cols, llvm::Value *index, llvm::Value *mask) {
  std::vector<llvm::Value *> value(g.size(), nullptr);
  std::map<std::pair<std::uint32_t, rt::NodeId>, typename Emitter<T>::Answers>
      called;
  for (const rt::Step &s : g.schedule()) {
    const auto &[v, c] = std::pair{s.node, s.fma};
    const auto &p = g[v];
    if (c) {
      value[v] = emit.fma(c, value[c.x], value[c.y], value[c.z]);
      continue;
    }
    const auto operands = g.operands(v);
    switch (rt::detail::on_tensors(p) ? rt::Kind::Tensor : rt::kind_of(p.op)) {
    case rt::Kind::Const:
      value[v] = emit.constant(p.value);
      break;
    case rt::Kind::Input:
      value[v] =
          emit.load(cols.inputs[rt::input_column(
                        g.inputs(), p.op, rt::detail::get<rt::Slot>(p).value)],
                    index, mask, "");
      break;
    case rt::Kind::Nary: {
      // The terms into the running value in list order, a product term of a
      // sum as the fma the sweep takes where the step is fused; the first
      // term is formed.
      const rt::OpCode op = rt::fold_op(p.op);
      llvm::Value *acc = nullptr;
      for (const rt::NodeId t : rt::arguments(g, v)) {
        const rt::Contraction prod =
            s.fused ? rt::product_of(g, t) : rt::Contraction{};
        if (!prod) {
          acc = acc ? emit.binary(op, acc, value[t]) : value[t];
        } else if (acc) {
          acc = emit.fma(prod, value[prod.x], value[prod.y], acc);
        } else {
          acc = emit.binary(rt::OpCode::Mul, value[prod.x], value[prod.y]);
          acc = prod.negated ? emit.unary(rt::OpCode::Neg, acc) : acc;
        }
      }
      value[v] = acc;
      break;
    }
    case rt::Kind::Extern: {
      // One call per (function, argument list), whichever component asked
      // first; the rest are loads off the same answer.
      const auto [fn, component] = rt::detail::get<rt::ExternRef>(p);
      auto [answers, fresh] = called.try_emplace({fn, operands[0]});
      if (fresh) {
        const auto args =
            rt::arguments(g, v) |
            std::views::transform([&value](rt::NodeId a) { return value[a]; }) |
            impl::to<std::vector<llvm::Value *>>();
        answers->second = emit.extern_call(g.externs()[fn], args);
      }
      value[v] = emit.component(answers->second, component);
      break;
    }
    case rt::Kind::Unary:
      value[v] = emit.unary(p.op, value[operands[0]]);
      break;
    case rt::Kind::Ternary:
      value[v] = emit.ternary(value[operands[0]], value[operands[1]],
                              value[operands[2]]);
      break;
    case rt::Kind::Binary:
      value[v] = emit.binary(p.op, value[operands[0]], value[operands[1]]);
      break;
    case rt::Kind::Tensor:
      if (g.fused_into(v) != rt::no_node) {
        break; // formed inside its reader
      }
      if (loops->handles(v)) {
        loops->emit(v, value);
        if (rt::detail::is_scalar(p)) {
          value[v] = emit.from_tape(g.rows()[v], mask);
        }
        break;
      }
      rt::detail::for_each_operand(g, v, [&](rt::NodeId u) {
        if (!rt::detail::on_tensors(g[u])) {
          emit.to_tape(value[u], g.rows()[u], mask);
        }
      });
      emit.tensor_step(v);
      if (rt::detail::is_scalar(p)) {
        value[v] = emit.from_tape(g.rows()[v], mask);
      }
      break;
    }
  }
  return value;
}

template <CKernelScalar T>
void emit_stores(const Emitter<T> &emit, const rt::Graph<T> &g,
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

// With tensor rows the outputs leave through the sweep's own scatter: each
// scalar into its tape row, then one call for every block.
template <CKernelScalar T>
void emit_tensor_stores(const Emitter<T> &emit, const rt::Graph<T> &g,
                        std::span<llvm::Value *const> value,
                        llvm::Function &kernel, llvm::Value *index,
                        llvm::Value *live, llvm::Value *mask) {
  for (const rt::NodeId o : g.outputs()) {
    if (!rt::detail::on_tensors(g[o])) {
      emit.to_tape(value[o], g.rows()[o], mask);
    }
  }
  emit.tensor_store(kernel, index, live);
}

} // namespace

template <CKernelScalar T>
std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<T> &g, llvm::StringRef name,
            const unsigned width, const llvm::DataLayout &layout,
            llvm::StringRef triple) {
  auto m = std::make_unique<llvm::Module>("ddx.jit", ctx);
  // Before the first instruction: IRBuilder reads the layout for every
  // alignment it fills in without being told one.
  m->setDataLayout(layout);
  m->setTargetTriple(triple);
  llvm::Function *const fn = declare_kernel(*m, name, g);
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::Type *const scalar = std::same_as<T, float>
                                 ? llvm::Type::getFloatTy(ctx)
                                 : llvm::Type::getDoubleTy(ctx);
  llvm::Argument *const count = fn->getArg(4);
  const LaneType lanes{
      .width = width,
      .scalar = scalar,
      .ty = width > 1 ? llvm::FixedVectorType::get(scalar, width) : scalar};

  auto *const entry = llvm::BasicBlock::Create(ctx, "entry", fn);
  auto *const loop = llvm::BasicBlock::Create(ctx, "loop", fn);
  auto *const exit = llvm::BasicBlock::Create(ctx, "exit", fn);

  // No fast-math flags at all: contraction is decided in the graph and spelled
  // llvm.fma, so allowContract would fuse a *second* set the sweep computed
  // separately.
  llvm::IRBuilder<> b(entry);
  const HoistedColumns cols = hoist_columns(b, *fn, g);
  // The scalar tape, tape_rows() rows `width` wide, then the tensor tape.
  const TensorTapes tensors =
      g.has_tensors() ? TensorTapes{.call = fn->getArg(5),
                                    .tape = fn->getArg(6),
                                    .vtape = b.CreateConstInBoundsGEP1_64(
                                        scalar, fn->getArg(6),
                                        g.tape_rows() * width, "vtape")}
                      : TensorTapes{};

  // An element of a tensor loop is one T, whatever the kernel's width.
  const Emitter<T> elem(
      *m, b, LaneType{.width = 1, .scalar = scalar, .ty = scalar}, tensors);
  std::optional<TensorLoops<T>> loops;
  if (g.has_tensors()) {
    loops.emplace(*m, b, g, elem, lanes, tensors);
  }

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
  llvm::Value *live = b.getInt64(1); // lanes this iteration fills
  if (lanes.vector()) {
    // nuw: i < n on every entry to the body, so the difference never wraps.
    llvm::Value *const remaining =
        b.CreateSub(count, index, "remaining", /*HasNUW=*/true);
    llvm::Value *const step =
        b.CreateStepVector(llvm::FixedVectorType::get(i64, width));
    mask = b.CreateICmpSLT(step, b.CreateVectorSplat(width, remaining), "mask");
    live = b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, remaining,
                                   b.getInt64(width), nullptr, "live");
  }

  const Emitter<T> emit(*m, b, lanes, tensors);
  const std::vector<llvm::Value *> value =
      emit_nodes(emit, loops ? &*loops : nullptr, g, cols, index, mask);
  if (g.has_tensors()) {
    emit_tensor_stores(emit, g, value, *fn, index, live, mask);
  } else {
    emit_stores(emit, g, cols, value, index, mask);
  }

  // nuw/nsw: n counts points the caller has allocated, so n + W is nowhere
  // near either bound.  It is what lets SCEV name an exact trip count.
  llvm::Value *const next =
      b.CreateAdd(index, llvm::ConstantInt::get(i64, width), "i.next",
                  /*HasNUW=*/true, /*HasNSW=*/true);
  index->addIncoming(next, b.GetInsertBlock());
  b.CreateCondBr(b.CreateICmpUGE(next, count), exit, loop);

  b.SetInsertPoint(exit);
  b.CreateRetVoid();

  return llvm::verifyFunction(*fn, &llvm::errs()) ? nullptr : std::move(m);
}

template std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &, const rt::Graph<double> &, llvm::StringRef,
            unsigned, const llvm::DataLayout &, llvm::StringRef);
template std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &, const rt::Graph<float> &, llvm::StringRef,
            unsigned, const llvm::DataLayout &, llvm::StringRef);

} // namespace ddx::jit::detail
