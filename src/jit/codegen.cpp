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

// The LLVM intrinsic for each Call opcode, or not_intrinsic for a libm call.
// Cached in a table because lookupIntrinsicID is too slow to call per node.
llvm::Intrinsic::ID intrinsic_for(rt::OpCode op) {
  static const std::array<llvm::Intrinsic::ID, rt::op_count> table = [] {
    std::array<llvm::Intrinsic::ID, rt::op_count> t{};
    for (const auto i : std::views::iota(0uz, rt::op_count)) {
      const auto &[form, name] = rt::spelling_of(static_cast<rt::OpCode>(i));
      llvm::SmallString<32> spelled{"llvm."};
      spelled += name;
      // Pow stays a libm call: the simplifier rewrites llvm.pow with constant
      // exponents into sqrt or 1/x, which differ from libm's pow by an ulp.
      t[i] = form == rt::Form::Call && i != std::to_underlying(rt::OpCode::Pow)
                 ? llvm::Intrinsic::lookupIntrinsicID(spelled)
                 : llvm::Intrinsic::not_intrinsic;
    }
    return t;
  }();
  return table[std::to_underlying(op)];
}

// Declares an external callee that never throws and always returns. Passing
// `effects` lets the optimiser move calls to it.
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

// The kernel handles `width` points per iteration, as T when width is 1 and as
// <width x T> otherwise. Only memory accesses and libm calls differ by width.
struct LaneType {
  unsigned width;
  llvm::Type *scalar;
  llvm::Type *ty;
  [[nodiscard]] constexpr bool vector() const noexcept { return width > 1; }
};

// The kernel arguments a tensor step needs: the TensorCall and the two
// interpreter tapes. All are null in a graph without tensors.
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

  // A constant of the lane type, splatted when the lane type is a vector.
  [[nodiscard]] llvm::Value *constant(double v) const {
    return llvm::ConstantFP::get(lanes_.ty, v);
  }

  // Inactive lanes load 1.0, which is inside every op's domain, so a scalarised
  // libm call on them cannot hit a pole. Their results are never stored.
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
    // Sign is u > 0 ? 1 : u < 0 ? -1 : u - u. Only ±0 and NaN reach u - u.
    llvm::Value *const zero = constant(0.0);
    return b_.CreateSelect(b_.CreateFCmpOGT(u, zero), constant(1.0),
                           b_.CreateSelect(b_.CreateFCmpOLT(u, zero),
                                           constant(-1.0),
                                           b_.CreateFSub(u, u)));
  }

  // A real llvm.fma, not llvm.fmuladd, which only permits fusion.
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
    // Yields 1.0 or 0.0, as compare_impl does. The compare is ordered, so a
    // NaN operand gives false.
    case rt::Form::Compare:
      return b_.CreateUIToFP(b_.CreateFCmp(op == rt::OpCode::Lt
                                               ? llvm::CmpInst::FCMP_OLT
                                               : llvm::CmpInst::FCMP_OLE,
                                           l, r),
                             lanes_.ty);
    case rt::Form::Call:
      return call(op, {l, r});
    // Max and Min use llvm.maximum and llvm.minimum, which propagate NaN as
    // the interpreter does; maxnum and minnum do not.
    case rt::Form::Special:
    case rt::Form::None:
      return b_.CreateBinaryIntrinsic(op == rt::OpCode::Max
                                          ? llvm::Intrinsic::maximum
                                          : llvm::Intrinsic::minimum,
                                      l, r);
    }
    std::unreachable();
  }

  // The results of one extern call, indexed by component * width + lane.
  using Answers = std::vector<llvm::Value *>;

  // Calls the extern once per lane, since it has only a scalar entry point.
  // One call returns the value and all of its partials for that lane.
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
  // Gathers one component of an extern call's results into the lane type.
  llvm::Value *component(const Answers &answers, std::uint32_t c) const {
    return gathered(
        [&](unsigned lane) { return answers[c * lanes_.width + lane]; });
  }

  llvm::Value *ternary(llvm::Value *c, llvm::Value *t, llvm::Value *f) const {
    // UNE, not ONE: a NaN condition counts as true, as C's != does in
    // select_impl. Only a raw value used as a condition can be NaN.
    return b_.CreateSelect(b_.CreateFCmpUNE(c, constant(0.0)), t, f);
  }

  // Lane k of tape row r lives at tape + r * width + k, as in the interpreter.
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
  // Calls a runtime entry point: the extern trampoline, the tensor step, or
  // the tensor store.
  void runtime_call(std::string_view symbol,
                    llvm::ArrayRef<llvm::Value *> args) const {
    const auto types = llvm::to_vector(
        llvm::map_range(args, [](llvm::Value *a) { return a->getType(); }));
    b_.CreateCall(
        declare(m_, libm_name<T>(symbol),
                llvm::FunctionType::get(b_.getVoidTy(), types, false)),
        args);
  }

  [[nodiscard]] llvm::Value *lanes_at(std::uint32_t row) const {
    return b_.getInt64(std::uint64_t{row} * lanes_.width);
  }

  // Intrinsics take the lane type directly; libm calls are made once per lane.
  // Callees are cached per opcode because declaring one per node is slow.
  llvm::Value *call(rt::OpCode op, llvm::ArrayRef<llvm::Value *> args) const {
    auto &[fn, scalarise] = callees_[std::to_underlying(op)];
    if (fn == nullptr) {
      const llvm::Intrinsic::ID id = intrinsic_for(op);
      if (id != llvm::Intrinsic::not_intrinsic) {
        fn = llvm::Intrinsic::getOrInsertDeclaration(&m_, id, {lanes_.ty});
      } else {
        // Not touching memory is the IR form of -fno-math-errno, so a call
        // can be hoisted.
        const llvm::SmallVector<llvm::Type *, 2> params(args.size(),
                                                        lanes_.scalar);
        fn = declare(m_, libm_name<T>(rt::spelling_of(op).name),
                     llvm::FunctionType::get(lanes_.scalar, params, false),
                     llvm::MemoryEffects::none());
        fn->setDoesNotFreeMemory();
        scalarise = lanes_.vector();
      }
    }
    return scalarise ? scalarised(fn, args) : libm_call(fn, args);
  }
  // nobuiltin stops the simplifier from replacing a libm call with another
  // function that rounds differently.
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

  // Calls `scalar` once per lane with that lane's element of each argument.
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
  // Builds a lane-type value from one scalar per lane.
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

// Emits tensor nodes as loop nests, columns outer and rows inner. The inner
// loop is force-vectorised, which may reorder a reduction's adds.
template <CKernelScalar T> class TensorLoops {
public:
  TensorLoops(llvm::Module &m, llvm::IRBuilder<> &b, const rt::Graph<T> &g,
              const Emitter<T> &elem, LaneType lanes, TensorTapes tapes)
      : m_{m}, b_{b}, g_{g}, elem_{elem}, lanes_{lanes}, tapes_{tapes},
        tables_(g.size(), nullptr), stepped_(g.size(), false),
        done_(g.size(), false) {
    // Table pointers are fetched once per call, in the entry block.
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
      // Operands of interpreter steps need a tape row; loops read tables in
      // place.
      if (!handles(v)) {
        rt::detail::for_each_operand(g, v,
                                     [&](rt::NodeId u) { stepped_[u] = true; });
      }
    }
    for (const rt::NodeId o : g.outputs()) {
      stepped_[o] = true;
    }
  }

  // MatMul and StackRows are left to the interpreter; the rest become loops.
  [[nodiscard]] bool handles(rt::NodeId v) const {
    return g_.op_of(v) != rt::OpCode::MatMul &&
           g_.op_of(v) != rt::OpCode::StackRows;
  }

  // Emits node v for every lane. A scalar result is left in its tape row.
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
  [[nodiscard]] std::uint64_t rows(rt::NodeId u) const {
    return g_.shape(u).rows;
  }

  // Emits for (iv = 0; iv < n; ++iv), threading `init` through `body`, which
  // returns the next carried values. Returns the final carried values.
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
      // Unrolling is disabled because a fully unrolled reduction becomes a
      // serial chain the vectoriser cannot see.
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
  // Visits every (i, j) of a shape, with i innermost.
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
  // Lane k's copy of node u's tensor row.
  [[nodiscard]] llvm::Value *row(rt::NodeId u) const {
    const rt::TensorRow &r = g_.tensors().rows[u];
    return element(tapes_.vtape, add(i64(std::uint64_t{r.at} * lanes_.width),
                                     b_.CreateMul(k_, i64(r.width))));
  }
  // Lane k of a scalar node, read from its SSA value: by the time a fused
  // reader runs, the tape row may belong to another node.
  [[nodiscard]] llvm::Value *scalar(rt::NodeId u) const {
    return lanes_.width == 1 ? value_[u]
                             : b_.CreateExtractElement(value_[u], k_);
  }
  void scalar_out(rt::NodeId v, llvm::Value *x) const {
    b_.CreateStore(
        x, element(tapes_.tape,
                   add(i64(std::uint64_t{g_.rows()[v]} * lanes_.width), k_)));
  }

  // Element (i, j) of node u, read from a table or u's row, or computed inline.
  llvm::Value *at(rt::NodeId u, llvm::Value *i, llvm::Value *j) {
    switch (g_.op_of(u)) {
    case rt::OpCode::VecConst:
      return read(tables_[u], i);
    case rt::OpCode::MatRow:
      if (const rt::NodeId a = g_.operands(u)[0];
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

  // Element (i, j) of an elementwise node, multiplying operands in the
  // interpreter's order.
  llvm::Value *expr(rt::NodeId v, llvm::Value *i, llvm::Value *j) {
    const auto in = g_.operands(v);
    using rt::detail::operands;
    using enum rt::OpCode;
    switch (g_.op_of(v)) {
    case Map:
      return elem_.unary(rt::detail::get<rt::OpCode>(g_[v]),
                         at(operands<Map>(in).vec, i, j));
    case Zip: {
      const auto [left, right] = operands<Zip>(in);
      return elem_.binary(rt::detail::get<rt::OpCode>(g_[v]), at(left, i, j),
                          at(right, i, j));
    }
    case Scale: {
      const auto [s, vec] = operands<Scale>(in);
      return b_.CreateFMul(scalar(s), at(vec, i, j));
    }
    case Outer: {
      const auto [left, right] = operands<Outer>(in);
      return b_.CreateFMul(at(right, j), at(left, i));
    }
    case RowScale: {
      const auto [scale, matrix] = operands<RowScale>(in);
      return b_.CreateFMul(at(scale, i), at(matrix, i, j));
    }
    case ColScale: {
      const auto [matrix, scale] = operands<ColScale>(in);
      return b_.CreateFMul(at(scale, j), at(matrix, i, j));
    }
    case VSelect: {
      const auto [cond, then, otherwise] = operands<VSelect>(in);
      return elem_.ternary(rt::detail::is_tensor(g_[cond]) ? at(cond, i, j)
                                                           : scalar(cond),
                           at(then, i, j), at(otherwise, i, j));
    }
    case Diag:
      return b_.CreateSelect(b_.CreateICmpEQ(i, j),
                             at(operands<Diag>(in).vec, i),
                             elem_.constant(0.0));
    case MatRow: { // a transpose, of a table or a computed matrix
      const rt::NodeId source = operands<MatRow>(in).source;
      return g_.op_of(source) == MatConst ? at(v, i, j) : at(source, j, i);
    }
    default:
      std::unreachable();
    }
  }

  llvm::Value *fmuladd(llvm::Value *x, llvm::Value *y, llvm::Value *z) const {
    return b_.CreateIntrinsic(llvm::Intrinsic::fmuladd, {lanes_.scalar},
                              {x, y, z});
  }

  // Computes M x into `plain` and M^T x into `transposed`, either optional,
  // four columns per pass so each element of M is loaded once for both.
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
      // A table is read in place. It is copied only when it is an output, or a
      // transpose whose row an interpreter step reads.
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
    {
      // The element index is flat; a fused matrix is read by row and column.
      const std::uint64_t e = rt::detail::get<rt::Element>(p).value;
      scalar_out(v, at(a, i64(e % rows(a)), i64(e / rows(a))));
      return;
    }
    case rt::OpCode::AddAt: {
      const auto [vec, add] = rt::detail::operands<rt::OpCode::AddAt>(g_.operands(v));
      llvm::Value *const out = row(v);
      each(s, {}, [&](llvm::Value *i, llvm::Value *j, auto) {
        b_.CreateStore(at(vec, i, j), element(out, flat(i, j, s.rows)));
        return Carried{};
      });
      llvm::Value *const to =
          element(out, i64(rt::detail::get<rt::Element>(p).value));
      b_.CreateStore(
          b_.CreateFAdd(b_.CreateLoad(lanes_.scalar, to), scalar(add)), to);
      return;
    }
    case rt::OpCode::Sum:
      scalar_out(
          v, each(g_.shape(a), {zero}, [&](auto i, auto j, const auto &acc) {
            return Carried{b_.CreateFAdd(acc[0], at(a, i, j))};
          })[0]);
      return;
    case rt::OpCode::Dot: {
      const auto [left, right] = rt::detail::operands<rt::OpCode::Dot>(g_.operands(v));
      scalar_out(
          v, each(g_.shape(left), {zero}, [&](auto i, auto j, const auto &acc) {
            return Carried{fmuladd(at(left, i, j), at(right, i, j), acc[0])};
          })[0]);
      return;
    }
    case rt::OpCode::MatVec: {
      // A paired M x and M^T x share one nest, emitted at whichever is first.
      const rt::NodeId mate = g_.pairs()[v];
      if (done_[v]) {
        return;
      }
      const bool transposed =
          rt::detail::get<rt::Orientation>(p) == rt::Orientation::transposed;
      if (mate != rt::no_node) {
        done_[mate] = true;
      }
      const auto [matrix, vec] = rt::detail::operands<rt::OpCode::MatVec>(g_.operands(v));
      products(matrix, vec, transposed ? mate : v, transposed ? v : mate);
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
    // Every remaining node is elementwise and gets a row of its own.
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
  std::vector<llvm::Value *> tables_; // per node, a table's elements
  std::vector<bool> stepped_;         // per node, read as a row
  std::vector<bool> done_;            // per node, emitted with its pair
  std::span<llvm::Value *const> value_;
  llvm::Value *k_ = nullptr;
};

template <CKernelScalar T>
llvm::Function *declare_kernel(llvm::Module &m, llvm::StringRef name,
                               const rt::Graph<T> &g) {
  llvm::LLVMContext &ctx = m.getContext();
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::PointerType *const ptr = llvm::PointerType::getUnqual(ctx);
  // Four column-pointer arrays, the batch length, then the tensor call and the
  // tape that tensor steps read.
  auto *const fty = llvm::FunctionType::get(
      llvm::Type::getVoidTy(ctx), {ptr, ptr, ptr, ptr, i64, ptr, ptr}, false);
  auto *const fn =
      llvm::Function::Create(fty, llvm::Function::ExternalLinkage, name, m);

  static constexpr std::array names{"xs", "f",       "g",    "h",
                                    "n",  "tensors", "tapes"};
  for (const auto [i, arg_name] : names | std::views::enumerate) {
    fn->getArg(static_cast<unsigned>(i))->setName(arg_name);
  }

  // These attributes describe the pointer arrays, not the columns they hold,
  // which are written through loaded pointers. Unrequested blocks may be null.
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

  // Without nounwind the optimiser cannot move code across the libm calls.
  fn->setDoesNotThrow();
  fn->setWillReturn();
  fn->setMustProgress();
  return fn;
}

// Column pointers loaded once in the entry block, so they are loop-invariant.
using ColumnValues = boost::container::small_vector<llvm::Value *, 8>;
struct HoistedColumns {
  ColumnValues inputs;
  // Value columns f[k], Jacobian g[k*n + j] and Hessian h[c*n + i].
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

// Node ids are topological, so one pass in schedule order suffices. Tensor
// nodes become loop nests, or interpreter steps that exchange scalars via tape.
template <CKernelScalar T>
[[nodiscard]] std::vector<llvm::Value *>
emit_nodes(const Emitter<T> &emit, TensorLoops<T> *loops, const rt::Graph<T> &g,
           const HoistedColumns &cols, llvm::Value *index, llvm::Value *mask) {
  std::vector<llvm::Value *> value(g.size(), nullptr);
  std::map<std::pair<std::uint32_t, rt::ListOffset>, typename Emitter<T>::Answers>
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
                        g.inputs(), p.op, rt::detail::get<rt::Slot>(p))],
                    index, mask, "");
      break;
    case rt::Kind::Nary: {
      // Folds the terms in list order. In a fused step a product term becomes
      // an fma, as the sweep computes it.
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
      // One call per function and argument list; other components reuse its
      // results. Lists are interned, so the offset identifies the list.
      const auto [fn, component] = rt::detail::get<rt::ExternRef>(p);
      auto [answers, fresh] = called.try_emplace({fn, g.list_of(v)});
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
      } else {
        rt::detail::for_each_operand(g, v, [&](rt::NodeId u) {
          if (!rt::detail::on_tensors(g[u])) {
            emit.to_tape(value[u], g.rows()[u], mask);
          }
        });
        emit.tensor_step(v);
      }
      if (rt::detail::is_scalar(p)) {
        value[v] = emit.from_tape(g.rows()[v], mask);
      }
      break;
    }
  }
  return value;
}

} // namespace

template <CKernelScalar T>
std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<T> &g, llvm::StringRef name,
            const unsigned width, const llvm::DataLayout &layout,
            llvm::StringRef triple) {
  auto m = std::make_unique<llvm::Module>("ddx.jit", ctx);
  // Set the data layout first, since IRBuilder reads it for default alignments.
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

  // No fast-math flags: contraction is decided in the graph and emitted as
  // llvm.fma, and allowContract would fuse pairs the sweep keeps apart.
  llvm::IRBuilder<> b(entry);
  const HoistedColumns cols = hoist_columns(b, *fn, g);
  // The tape argument holds the scalar tape, tape_rows() rows of `width`, and
  // then the tensor tape.
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

  // Weighting n == 0 as unlikely keeps the loop out of the cold code layout.
  llvm::MDBuilder md(ctx);
  b.CreateCondBr(b.CreateICmpEQ(count, llvm::ConstantInt::get(i64, 0)), exit,
                 loop, md.createBranchWeights(1, 1U << 12));

  b.SetInsertPoint(loop);
  llvm::PHINode *const index = b.CreatePHI(i64, 2, "i");
  index->addIncoming(llvm::ConstantInt::get(i64, 0), entry);

  // Lane k is live while k < n - i. The signed compare is safe since both sides
  // are non-negative, and it is far cheaper than llvm.get.active.lane.mask.
  llvm::Value *mask = nullptr;
  llvm::Value *live = b.getInt64(1); // lanes filled this iteration
  if (lanes.vector()) {
    // nuw holds because i < n whenever the body runs.
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
    // With tensors, outputs go through the interpreter's scatter: scalars are
    // stored to their tape rows, then one call writes every block.
    for (const rt::NodeId o : g.outputs()) {
      if (!rt::detail::on_tensors(g[o])) {
        emit.to_tape(value[o], g.rows()[o], mask);
      }
    }
    emit.tensor_store(*fn, index, live);
  } else {
    const rt::PerBlock<std::span<llvm::Value *const>> columns{
        cols.out.values, cols.out.jacobian, cols.out.hessian};
    for (const auto [block_columns, block] :
         rt::zip_blocks(columns, g.output_blocks())) {
      for (const auto [column, o] : std::views::zip(block_columns, block)) {
        emit.store(value[o], column, index, mask);
      }
    }
  }

  // nuw and nsw hold because n counts allocated points, and they let SCEV
  // compute an exact trip count.
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
