#pragma once

#include "columns.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/archive/archive.hpp"
#include "rt/cache.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/dot.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "rt/lanes.hpp"
#include "util/ranges.hpp"

// Unconditional: jit::Options and jit::Kernel are header types, so the Python
// surface has one shape whether or not the backend was compiled in.
#include "jit/kernel.hpp"
#include "rt/equation.hpp"

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <future>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

// What ddx::impl::Equation is, with the output count a runtime value.  Much
// smaller than the facade because the GIL is the lock: every freeze, poll and
// adopt below runs holding it, and only the kernel or the sweep gives it up.
namespace ddx::py {

class PyCall;

// The compressed Hessian back into a dense n x n a caller reads, with the zeros
// the colouring left out put back: one fill, then one copy per cell a column
// owns.  Indexed through Coloring::entries, never as a dense colours x n grid
// -- the block holds only the owned cells, so `cells()` is below count() * n
// wherever the pattern is sparse and the dense reading runs off the end.
inline void scatter_hessian(const rt::Coloring &coloring,
                            std::span<double *const> cells, double *dense,
                            std::size_t n, std::size_t points) {
  const auto wide = static_cast<std::ptrdiff_t>(points);
  std::ranges::fill_n(dense, static_cast<std::ptrdiff_t>(n * n) * wide, 0.0);
  for (const rt::Cell &cell : coloring.entries()) {
    std::ranges::copy_n(cells[cell.slot], wide,
                        dense + (cell.row * n + cell.column) * points);
  }
}

// A Python caller cannot name a C++ type, so the only choice is whether there
// is one.
using PyCache = rt::LastValue<double>;
class PyEquation : public rt::detail::Caching<PyEquation, double, PyCache> {
public:
  using Want = rt::Want;
  using Memo = rt::detail::Caching<PyEquation, double, PyCache>;
  // Reaches arena() and unlocked(); neither is anyone else's business.
  friend Memo;

  PyEquation(std::shared_ptr<rt::Builder<double>> arena,
             std::vector<rt::NodeId> roots, bool remember = false)
      : arena_{std::move(arena)}, symbols_{arena_->symbols()},
        roots_{std::move(roots)},
        model_nodes_{static_cast<std::uint32_t>(arena_->size())} {
    take_cache(PyCache{remember});
  }

  // From a file: the sweeps arrive filled, so nothing below computes them.
  explicit PyEquation(rt::Verified<double> snap, bool remember = false)
      : PyEquation{std::move(snap).rebuild(), remember} {}
  explicit PyEquation(rt::Rebuilt<double> r, bool remember = false)
      : arena_{std::move(r.arena)}, symbols_{arena_->symbols()},
        roots_{std::move(r.rest.roots)},
        derivative_{std::move(r.rest.jacobian)},
        sweeps_{std::move(r.rest.hessians)}, options_{r.rest.options},
        objects_{std::move(r.rest.objects)}, model_nodes_{r.rest.model_nodes},
        loaded_{true} {
    settle_device();
    take_cache(PyCache{remember});
  }

  // The same struct the C++ facade fills: one serialiser, two equations.  Also
  // what the cache path in module.cpp writes, which is why it is not private.
  [[nodiscard]] rt::Snapshot<double> snapshot() {
    rt::Snapshot<double> snap;
    // The sweeps first, the arena after: they *append*, so a node array taken
    // before they run lacks the ids they hand back.  The order matters here and
    // not in the facade because this side sweeps lazily.
    snap.jacobian = derivative();
    snap.hessians = sweeps();
    snap.symbols = arena_->symbols();
    snap.nodes.assign(arena_->nodes().begin(), arena_->nodes().end());
    snap.roots = roots_;
    snap.options = options_;
    snap.model_nodes = model_nodes_;
#ifdef DDX_HAS_JIT
    snap.objects = objects();
#endif
    return snap;
  }

  [[nodiscard]] constexpr bool loaded() const noexcept { return loaded_; }

  [[nodiscard]] std::size_t arity() const { return arena_->symbols().size(); }
  [[nodiscard]] constexpr std::size_t outputs() const { return roots_.size(); }

  [[nodiscard]] const std::vector<std::string> &symbols() const {
    return arena_->symbols();
  }
  // The Python spelling: a list over the keys Symbols already built, so an
  // access costs increfs and never a PyUnicode.
  [[nodiscard]] pyb::list symbols_list() const { return symbols_.listed(); }

  [[nodiscard]] pyb::object evaluate(const pyb::handle &x) {
    const Point at{x, symbols_};
    const auto l = lane(Want::Value);
    if (scalar(at)) {
      Cell f;
      run(Want::Value, at, {.values = f.rows()});
      return pyb::float_(f.value);
    }
    Block f{shape_of({ssize(outputs())}, at),
            count(*l, &rt::OutputSpans::values), at.size()};
    run(Want::Value, at, {.values = f.rows()});
    return std::move(f).array();
  }

  [[nodiscard]] pyb::tuple jacobian(const pyb::handle &x) {
    const Point at{x, symbols_};
    const auto l = lane(Want::Jacobian);
    Scratch cells{count(*l, &rt::OutputSpans::jacobian), at.size()};
    Dense g{shape_of({ssize(outputs()), ssize(arity())}, at), at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run(Want::Jacobian, at,
        {.values = f ? f->rows() : cell.rows(), .jacobian = cells.rows()});
    scatter_jacobian(cells, g, at.size());
    return pyb::make_tuple(value_of(f, cell), std::move(g).array());
  }

  // The Jacobian alone, from a lane with no value block.
  [[nodiscard]] pyb::object gradient(const pyb::handle &x) {
    const Point at{x, symbols_};
    const auto l = lane(Want::Gradient);
    Scratch cells{count(*l, &rt::OutputSpans::jacobian), at.size()};
    Dense g{shape_of({ssize(outputs()), ssize(arity())}, at), at.size()};
    run(Want::Gradient, at, {.jacobian = cells.rows()});
    scatter_jacobian(cells, g, at.size());
    return std::move(g).array();
  }

  [[nodiscard]] pyb::tuple hessian(const pyb::handle &x) {
    const Point at{x, symbols_};
    return outputs() == 1 ? hessian_from_lane(at) : hessian_from_arena(at);
  }

  [[nodiscard]] pyb::tuple jvp(const pyb::handle &v, const pyb::handle &x) {
    const Point at{x, symbols_};
    const Point dir{v, symbols_};
    const auto l = lane(Want::Jvp);
    Block out{shape_of({ssize(outputs())}, at),
              count(*l, &rt::OutputSpans::jacobian), at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run_seeded(Want::Jvp, at, dir,
               {.values = f ? f->rows() : cell.rows(), .jacobian = out.rows()});
    return pyb::make_tuple(value_of(f, cell), std::move(out).array());
  }

  [[nodiscard]] pyb::tuple vjp(const pyb::handle &w, const pyb::handle &x) {
    const Point at{x, symbols_};
    // By position: the covector is indexed by function, and a dict would have
    // nothing to key on.
    const Point dir{w, outputs()};
    const auto l = lane(Want::Vjp);
    Block out{symbol_shape(at), count(*l, &rt::OutputSpans::jacobian),
              at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run_seeded(Want::Vjp, at, dir,
               {.values = f ? f->rows() : cell.rows(), .jacobian = out.rows()});
    return pyb::make_tuple(value_of(f, cell), std::move(out).array());
  }

  // (value, gradient, H v): the gradient comes off the same sweep.  One
  // output only, as hessian() is.
  [[nodiscard]] pyb::tuple hvp(const pyb::handle &v, const pyb::handle &x) {
    if (outputs() != 1) {
      fail_with(errc::not_univariate);
    }
    const Point at{x, symbols_};
    const Point dir{v, symbols_};
    const auto l = lane(Want::Hvp);
    const std::size_t n = arity();

    Scratch partials{count(*l, &rt::OutputSpans::jacobian), at.size()};
    Dense g{shape_of({ssize(outputs()), ssize(n)}, at), at.size()};
    Block out{symbol_shape(at), count(*l, &rt::OutputSpans::hessian), at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run_seeded(Want::Hvp, at, dir,
               {.values = f ? f->rows() : cell.rows(),
                .jacobian = partials.rows(),
                .hessian = out.rows()});
    scatter_jacobian(partials, g, at.size());
    return pyb::make_tuple(value_of(f, cell), std::move(g).array(),
                           std::move(out).array());
  }

  [[nodiscard]] constexpr const jit::Options &options() const noexcept {
    return options_;
  }

  // Discards whatever was compiled, so the choice holds however late.  Choosing
  // a compiling backend starts the compile, so it overlaps what comes next.
  void set_options(const jit::Options &opt) {
    if (opt == options_) {
      return;
    }
    options_ = opt;
    settle_device();
    lanes_ = {};
    // codegen decides the arithmetic, so what was remembered was answered by
    // another graph.
    cache().clear();
    if (opt.backend != jit::Backend::Interpret) {
      (void)lane(Want::Jacobian);
    }
  }

  // The only thing that waits.  A call does not: until the kernel lands the
  // graph is swept, and the kernel replaces the sweep when it arrives.
  [[nodiscard]] bool wait_for_kernel(Want want) {
    (void)lane(want); // launches it, if nothing has yet
    if (const auto first = lane_for(want).pending(); first.valid()) {
      // Given up around the wait alone, never around the lane bookkeeping.
      const pyb::gil_scoped_release unlocked;
      first.wait();
    }
    return static_cast<bool>(lane(want)->kernel);
  }

  [[nodiscard]] bool uses_kernel() {
    return static_cast<bool>(lane(Want::Jacobian)->kernel);
  }

  // Under Backend.DEVICE the device answering, None under any other backend.
  // Raises with the reason where no device answers or its compiler refused.
  [[nodiscard]] std::optional<std::string> device_status() {
    if (options_.backend != jit::Backend::Device) {
      return std::nullopt;
    }
#ifdef DDX_HAS_OPENCL
    const auto &device = cl::Device::shared(options_.device);
    if (!device) {
      throw PyError{error{device.error().code}, device.error().detail};
    }
    (void)lane(Want::Jacobian); // adopts a compile that has landed
    if (const auto why = lane_for(Want::Jacobian).refused()) {
      throw PyError{error{why->code}, why->detail};
    }
    return std::string{device->identity()};
#else
    fail_with(errc::no_device);
#endif
  }

  // Colours, not n: the Hessian is stored compressed and scattered on read, so
  // this is what the compiled lane actually computes per point.
  [[nodiscard]] std::size_t hessian_colors() {
    return sweeps().front().colors();
  }

  // What one call for `want` evaluates: the contracted live order, which is
  // the sweep and what the kernel is lowered from.
  [[nodiscard]] std::size_t nodes(Want want) {
    return lane(want)->graph->schedule().size();
  }

  // Only what is free to answer: uses_kernel() would freeze a lane, and a repr
  // that compiles something is one nobody can call in a debugger.  Boost's
  // join, not views::join_with: libstdc++ 14.2 gets that one wrong under
  // clang, and 14.2 is what Ubuntu 24.04 ships.
  [[nodiscard]] std::string repr() const {
    return std::format("<ddx.Equation ({}) -> {} output{}>",
                       boost::algorithm::join(symbols(), ", "), outputs(),
                       outputs() == 1 ? "" : "s");
  }

  // the equation on disk
  void save(const std::filesystem::path &path) {
    unwrap(rt::save(snapshot(), path));
  }

  // Raises rather than answering false: "no" has three reasons -- unreadable,
  // unloadable, a different equation -- and only the errc says which.
  void verify(const std::filesystem::path &path) {
    const auto snap = rt::load_snapshot<double>(path);
    if (!snap) {
      throw PyError{snap.error()};
    }
    if (!describes(**snap)) {
      fail_with(errc::archive_mismatch);
    }
  }

  // The roots as well as the model: two equations over one arena share every
  // node and differ only in which ids they call outputs.
  [[nodiscard]] bool describes(const rt::Snapshot<double> &snap) const {
    return snap.roots.size() == roots_.size() &&
           std::ranges::equal(snap.roots, roots_) &&
           rt::digest<double>(snap.symbols, snap.nodes, snap.model_nodes) ==
               rt::digest<double>(arena_->symbols(), arena_->nodes(),
                                  model_nodes_);
  }

  [[nodiscard]] std::string to_dot(bool all) {
    return rt::Dot<double>{*lane(Want::Jacobian)->graph,
                           all ? rt::Scope::All : rt::Scope::Live}
        .str();
  }

private:
  friend class PyCall;

#ifdef DDX_HAS_JIT
  // The machine code the lanes are holding, for the file to carry.  Only
  // kernels that kept their bytes -- Options.retain_object, on by default.
  [[nodiscard]] std::vector<rt::Object> objects() {
    jit::Compiler *const c = compiler();
    if (c == nullptr) {
      return {};
    }
    std::vector<rt::Object> out;
    for (const auto [want, l] : std::views::zip(rt::want_values, lanes_)) {
      if (auto kept = l.object(want, setting())) {
        out.push_back(*std::move(kept));
      }
    }
    return out;
  }
#endif

  // One rung, because the Python surface has no ladder to climb: warm_points is
  // the only threshold it has and hot_points says nothing.  No lock, because
  // the GIL is held for every touch below and given up only inside a sweep.
  using Lane = rt::detail::Lane<double, rt::detail::Serialised, 1>;
  using Compiled = rt::detail::Compiled<double>;

  [[nodiscard]] Lane &lane_for(Want want) {
    return lanes_[static_cast<std::size_t>(want)];
  }

  // `n` is the batch about to run and is what Adapt counts; an observer asks
  // without buying anything.
  [[nodiscard]] std::shared_ptr<const Compiled> lane(Want want,
                                                     std::size_t n = 0) {
    return lane_for(want).snapshot(want, n, setting(),
                                   [&] { return frozen_for(want); });
  }

  [[nodiscard]] rt::detail::Setting setting() {
    rt::detail::Setting out;
    out.options = effective_options();
    out.objects = objects_;
#ifdef DDX_HAS_JIT
    if (options_.backend == jit::Backend::Compile ||
        options_.backend == jit::Backend::Adapt) {
      out.compiler = compiler();
    }
#endif
#ifdef DDX_HAS_OPENCL
    out.device = device_;
#endif
    return out;
  }

  // Looked up where the options change, so a call never resolves a selector.
  void settle_device() {
#ifdef DDX_HAS_OPENCL
    device_ = options_.backend == jit::Backend::Device
                  ? impl::rt_detail::shared_device(options_.device)
                  : nullptr;
#endif
  }

  // What a freeze asks for.  Every one of these sweeps on first call and
  // appends to the arena, so they are reached through here and not gathered up
  // front: a lane is frozen from the one block its want names.
  struct Frozen {
    PyEquation &eq;

    [[nodiscard]] std::span<const rt::NodeId> roots() const {
      return eq.roots_;
    }
    [[nodiscard]] const rt::Jacobian &jacobian() const {
      return eq.derivative();
    }
    [[nodiscard]] const rt::Hessian &hessian() const {
      return eq.sweeps().front();
    }
    [[nodiscard]] const rt::HessianVector &hessian_vector() const {
      return eq.hvp_sweep();
    }
    [[nodiscard]] const rt::VectorJacobian &vector_jacobian() const {
      return eq.vjp_sweep();
    }
    [[nodiscard]] const rt::Tangent &tangent() const { return eq.jvp_sweep(); }
  };

  // This side never rebalances, so the kernel is lowered from the graph the
  // sweep walks and the two blocks alias.
  [[nodiscard]] Compiled frozen_for(Want want) {
    Frozen from{*this};
    auto swept = std::make_shared<const rt::Graph<double>>(
        rt::freeze_for(want, *arena_, from, contracts()));
    return {.graph = swept, .compile_graph = swept};
  }

  // How wide to emit, from the batch the caller said they have: a wide kernel
  // does a register's worth of work to answer for a single point.
  [[nodiscard]] constexpr jit::Options effective_options() const noexcept {
    return jit::for_batch(options_, rt::block_lanes);
  }

#ifdef DDX_HAS_JIT
  [[nodiscard]] static jit::Compiler *compiler() {
    return impl::rt_detail::shared_compiler();
  }
#endif

  // Decided in the graph, so a batch answers the same whichever path ran it.
  [[nodiscard]] constexpr bool contracts() const noexcept {
    return options_.codegen.contract;
  }

  using BlockMember = std::span<const rt::NodeId> rt::OutputSpans::*;
  [[nodiscard]] static std::size_t count(const Compiled &c,
                                         BlockMember which) {
    return (c.graph->output_blocks().*which).size();
  }

  // --- running -------------------------------------------------------------

  void run(Want want, const Point &at, const rt::Columns<double> &out) {
    run_columns(want, at.columns(), at.size(), out);
  }

  // The point's columns and the direction's, in one array: symbols first, then
  // the seeds, which is the order input_column() states.
  void run_seeded(Want want, const Point &at, const Point &dir,
                  const rt::Columns<double> &out) {
    if (dir.size() != at.size()) {
      fail_with(errc::wrong_direction);
    }
    boost::container::small_vector<const double *, 64> xs;
    xs.reserve(at.columns().size() + dir.columns().size());
    std::ranges::copy(at.columns(), std::back_inserter(xs));
    std::ranges::copy(dir.columns(), std::back_inserter(xs));
    run_columns(want, {xs.data(), xs.size()}, at.size(), out);
  }

  void run_columns(Want want, std::span<const double *const> xs, std::size_t n,
                   const rt::Columns<double> &out) {
    const auto l = lane(want, n);
    if (xs.size() != l->graph->arity() ||
        std::ranges::any_of(rt::zip_blocks(out, l->graph->output_blocks()),
                            [](const auto &pair) {
                              const auto &[columns, block] = pair;
                              return columns.size() != block.size();
                            })) {
      fail_with(errc::wrong_column_count);
    }
    through(want, *l->graph,
            l->kernel ? rt::Answered::ByKernel : rt::Answered::BySweep, xs, out,
            n, [&] { sweep(*l, xs, out, n); });
  }

  void sweep(const Compiled &l, std::span<const double *const> xs,
             const rt::Columns<double> &out, std::size_t n) {
    // Nothing here touches Python, so the GIL goes -- which is what lets
    // another thread call in while a long batch runs.
    const pyb::gil_scoped_release unlocked;
    // A kernel that could not run this call hands it back to the sweep.
    if (l.kernel(xs, out.values, out.jacobian, out.hessian, n)) {
      return;
    }
    interpret(*l.graph, xs, out, n);
  }

  // What Caching walks, and what it gives up around a sweep.
  [[nodiscard]] const rt::Builder<double> &arena() const { return *arena_; }
  // Named, not `return {}`: gil_scoped_release's bool constructor is explicit,
  // so a braced return would be reading the empty list as `false`.
  [[nodiscard]] static pyb::gil_scoped_release unlocked() {
    return pyb::gil_scoped_release{};
  }

  // The block sweep, within ~1.4x of a kernel and not a sad path.
  void interpret(const rt::Graph<double> &graph,
                 std::span<const double *const> xs,
                 const rt::Columns<double> &out, std::size_t n) const {
    const auto blocks = graph.output_blocks();
    const auto schedule = graph.schedule();
    // The graph's width, not the symbol count: `xs` carries one column per
    // input, and a seeded lane holds leaves that are not symbols.
    const std::size_t symbols = graph.arity();
    const auto scatter = [&](const auto &tape, std::size_t i,
                             std::size_t stride, std::size_t width) {
      rt::scatter_blocks(blocks, std::span<const double>{tape}, out, i, stride,
                         width);
    };

    // Tapes are sized by the arena rather than the graph: it may have grown
    // since the freeze, and live ids index below that.  A short batch sweeps
    // one point at a time rather than padding out to rt::block_lanes.
    //
    // thread_local rather than a member: run() drops the GIL, so two Python
    // threads can be inside one Equation at once and a shared buffer would be a
    // race.  Grown and never shrunk, so a loop allocates once.
    //
    // resize() rather than assign(): every id read is written first, so no
    // stale value is ever seen and zeroing the arena per call is waste.
    if (n < rt::block_lanes) {
      static thread_local std::vector<double> at;
      static thread_local std::vector<double> tape;
      at.resize(symbols);
      tape.resize(arena_->size());
      for (const std::size_t i : std::views::iota(0uz, n)) {
        std::ranges::transform(xs, at.begin(),
                               [i](const double *column) { return column[i]; });
        rt::evaluate_into(*arena_, at, schedule, std::span<double>{tape});
        scatter(tape, i, 1, 1);
      }
      return;
    }

    // rt::block_lanes points per sweep: the switch is paid once per node per
    // block, and each operation becomes a lane loop wide enough to vectorise.
    // A short final block repeats its last point, and those lanes are never
    // read.
    static thread_local std::vector<double> lanes;
    static thread_local std::vector<double> tape;
    lanes.resize(symbols * rt::block_lanes);
    tape.resize(arena_->size() * rt::block_lanes);

    for (const std::size_t base :
         std::views::iota(0uz, n) | std::views::stride(rt::block_lanes)) {
      const std::size_t width = std::min(rt::block_lanes, n - base);
      for (const auto [dst, column] :
           std::views::zip(lanes | std::views::chunk(rt::block_lanes), xs)) {
        const auto tail = std::ranges::copy_n(
            column + base, static_cast<std::ptrdiff_t>(width), dst.begin());
        std::ranges::fill(tail.out, dst.end(), column[base + width - 1]);
      }
      rt::evaluate_block<rt::block_lanes>(*arena_,
                                          std::span<const double>{lanes},
                                          schedule, std::span<double>{tape});
      scatter(tape, base, rt::block_lanes, width);
    }
  }

  // This side always hands back the dense matrix, so the zeros go back in
  // here: one fill, then one copy per cell the pattern names.
  void scatter_jacobian(Scratch &cells, Dense &dense, std::size_t points) {
    const std::size_t n = arity();
    const std::size_t wide = outputs() * n;
    if (wide == 0) {
      return;
    }
    std::ranges::fill_n(dense.at(0), static_cast<std::ptrdiff_t>(wide * points),
                        0.0);
    for (const auto [i, j, src] : derivative().pattern.cells(cells.rows())) {
      std::ranges::copy_n(src, static_cast<std::ptrdiff_t>(points),
                          dense.at(i * n + j));
    }
  }

  // --- the Hessian ---------------------------------------------------------

  // One function: the lane holds it compressed by colour, and the colouring
  // says which (colour, row) column owns a given (i, j).
  [[nodiscard]] pyb::tuple hessian_from_lane(const Point &at) {
    const auto l = lane(Want::Hessian);
    const std::size_t n = arity();
    Dense g{shape_of({ssize(outputs()), ssize(n)}, at), at.size()};
    Scratch partials{count(*l, &rt::OutputSpans::jacobian), at.size()};
    Scratch compressed{count(*l, &rt::OutputSpans::hessian), at.size()};
    Cell cell;
    auto f = values_block(*l, at);
    run(Want::Hessian, at,
        {.values = f ? f->rows() : cell.rows(),
         .jacobian = partials.rows(),
         .hessian = compressed.rows()});
    scatter_jacobian(partials, g, at.size());

    Dense dense{shape_of({ssize(outputs()), ssize(n), ssize(n)}, at),
                at.size()};
    scatter_hessian(l->graph->coloring(), compressed.rows(), dense.at(0), n,
                    at.size());
    return pyb::make_tuple(value_of(f, cell), std::move(g).array(),
                           std::move(dense).array());
  }

  // A system: one sweep per root, read off the arena.  A frozen graph carries a
  // single colouring, so no lane could hold m of these -- which is why this is
  // a point at a time and says so.
  [[nodiscard]] pyb::tuple hessian_from_arena(const Point &at) {
    if (at.batched() && at.size() != 1) {
      fail_with(errc::wrong_column_count);
    }
    const std::size_t n = arity();
    const auto &blocks = sweeps();

    boost::container::small_vector<double, 16> point(n);
    std::ranges::transform(at.columns(), point.begin(),
                           [](const double *c) { return *c; });
    // The schedule over what the three blocks below read -- values, partials
    // and Hessian::at's compressed cells -- settled on first ask: it depends
    // only on the sweeps, and the GIL is held here.  The tape is thread-kept;
    // every id read below is scheduled, so it is written this call.
    if (!hessian_schedule_) {
      auto wanted = blocks | std::views::transform(&rt::Hessian::compressed) |
                    std::views::join | impl::to<std::vector<rt::NodeId>>();
      impl::append(wanted, blocks |
                               std::views::transform(&rt::Hessian::partial) |
                               std::views::join);
      impl::append(wanted, blocks | std::views::transform(&rt::Hessian::zero));
      impl::append(wanted, roots_);
      const auto live = rt::detail::reachable(*arena_, wanted);
      hessian_schedule_ =
          rt::detail::schedule_of(*arena_, rt::detail::live_ids(live));
    }
    static thread_local std::vector<double> values;
    values.resize(arena_->size());
    rt::evaluate_into(*arena_, point, *hessian_schedule_,
                      std::span<double>{values});

    Dense f{shape_of({ssize(outputs())}, at), at.size()};
    Dense g{shape_of({ssize(outputs()), ssize(n)}, at), at.size()};
    Dense dense{shape_of({ssize(outputs()), ssize(n), ssize(n)}, at),
                at.size()};
    const auto dims = std::views::iota(0uz, n);
    for (const auto [k, block] : std::views::enumerate(blocks)) {
      const auto root = static_cast<std::size_t>(k);
      *f.at(root) = values[roots_[root]];
      for (const std::size_t i : dims) {
        *g.at(root * n + i) = values[block.partial[i]];
      }
      for (const auto [i, j] : std::views::cartesian_product(dims, dims)) {
        *dense.at((root * n + i) * n + j) = values[block.at(i, j)];
      }
    }
    return pyb::make_tuple(std::move(f).array(), std::move(g).array(),
                           std::move(dense).array());
  }

  // Swept once and kept.  Hash consing makes a repeat sweep add no nodes, so
  // this is about the colouring's cost, not the arena's size.
  [[nodiscard]] const std::vector<rt::Hessian> &sweeps() {
    if (!sweeps_) {
      sweeps_ = roots_ | std::views::transform([this](rt::NodeId r) {
                  return rt::build_hessian_impl(*arena_, r);
                }) |
                impl::to<std::vector<rt::Hessian>>();
    }
    return *sweeps_;
  }

  // The three seeded products, each swept on first ask: nothing is swept for a
  // caller who never asks.
  [[nodiscard]] const rt::HessianVector &hvp_sweep() {
    if (!hvp_) {
      hvp_ = rt::build_hvp_impl(*arena_, roots_.front());
    }
    return *hvp_;
  }
  [[nodiscard]] const rt::VectorJacobian &vjp_sweep() {
    if (!vjp_) {
      vjp_ = rt::build_vjp_impl(*arena_, roots_);
    }
    return *vjp_;
  }
  [[nodiscard]] const rt::Tangent &jvp_sweep() {
    if (!jvp_) {
      jvp_ = rt::build_jvp_impl(*arena_, roots_);
    }
    return *jvp_;
  }

  // Held for the same reason the Hessians are: swept per lane otherwise, which
  // leaves a loaded equation recomputing what its file just handed it.
  [[nodiscard]] const rt::Jacobian &derivative() {
    if (!derivative_) {
      derivative_ = rt::build_jacobian_impl(*arena_, roots_);
    }
    return *derivative_;
  }

  // --- shapes --------------------------------------------------------------

  // The leading axis goes at one function and the trailing one at one point, so
  // a scalar model at a point answers with a float, an (n,) gradient and an
  // (n, n) Hessian rather than three arrays with 1s in them.  Computed before
  // the array is built, so the array is built at this shape and never reshaped.
  [[nodiscard]] Shape shape_of(std::initializer_list<pyb::ssize_t> base,
                               const Point &at) const {
    const std::span axes{base};
    // A count and not a flag: drop() takes a difference_type, and libstdc++
    // 14.2 refuses the bool under clang where 14.3 accepts it.
    const auto lead =
        static_cast<std::ptrdiff_t>(outputs() == 1 && !axes.empty());
    Shape dims;
    std::ranges::copy(axes | std::views::drop(lead), std::back_inserter(dims));
    if (at.batched()) {
      dims.push_back(ssize(at.size()));
    }
    return dims;
  }

  // A block indexed by symbol rather than by function, so shape_of's leading
  // axis -- which it drops at one output -- is not one this has.
  [[nodiscard]] Shape symbol_shape(const Point &at) const {
    Shape dims{ssize(arity())};
    if (at.batched()) {
      dims.push_back(ssize(at.size()));
    }
    return dims;
  }

  // Exactly the case where the values block is one double and the answer is a
  // Python float, so it needs no array at all.
  [[nodiscard]] bool scalar(const Point &at) const {
    return outputs() == 1 && !at.batched();
  }

  // One double on the stack for that case.  The ABI wants a row pointer, and
  // this is the whole of what it writes.
  struct Cell {
    double value{};
    double *row{&value};
    [[nodiscard]] constexpr std::span<double *const> rows() {
      return {&row, 1};
    }
  };

  // A float at one point, an array across a batch: only the first can skip the
  // allocation, and every derivative call serves both.
  [[nodiscard]] std::optional<Block> values_block(const Compiled &c,
                                                  const Point &at) {
    return scalar(at) ? std::nullopt
                      : std::optional<Block>{
                            std::in_place, shape_of({ssize(outputs())}, at),
                            count(c, &rt::OutputSpans::values), at.size()};
  }

  [[nodiscard]] static pyb::object value_of(std::optional<Block> &f,
                                            Cell &cell) {
    return f ? std::move(*f).array() : pyb::object{pyb::float_(cell.value)};
  }

  std::shared_ptr<rt::Builder<double>> arena_;
  Symbols symbols_;
  std::vector<rt::NodeId> roots_;
  std::optional<rt::Jacobian> derivative_;
  std::optional<std::vector<rt::Hessian>> sweeps_;
  // A system's hessian() walk, settled on first ask beside the sweeps it reads.
  std::optional<std::vector<rt::Step>> hessian_schedule_;
  std::optional<rt::HessianVector> hvp_;
  std::optional<rt::VectorJacobian> vjp_;
  std::optional<rt::Tangent> jvp_;
  jit::Options options_{};
  // Machine code a file handed over, consulted once per lane by prepare().
  std::vector<rt::Object> objects_{};
#ifdef DDX_HAS_OPENCL
  // Settled with options_: null unless the backend is Device and one answered.
  const cl::Device *device_ = nullptr;
#endif
  // Where the model ends and the sweeps begin; set by make_equation once the
  // roots are in.
  std::uint32_t model_nodes_ = 0;
  bool loaded_ = false;
  std::array<Lane, rt::want_count> lanes_;
};

// A call bound to its buffers: the point's columns, the output arrays and the
// row pointers are taken once, so a repeat is a lane lookup and the sweep.
// Owning them rather than binding the caller's is what leaves no shape, dtype
// or stride to check.
class PyCall {
public:
  // The equation outlives the call through pyb::keep_alive on the factory,
  // never through a member: the binding states the lifetime once.
  PyCall(PyEquation &eq, PyEquation::Want want, const pyb::handle &x)
      : eq_(&eq), want_(want), x_(pyb::cast<Array>(x)), at_(x_, eq.symbols_) {
    // A system's Hessian is read off the arena a point at a time, so there is
    // no lane to bind.
    if (want_ == PyEquation::Want::Hessian && eq_->outputs() != 1) {
      fail_with(errc::wrong_column_count);
    }
    const auto l = eq_->lane(want_);
    const auto outputs = ssize(eq_->outputs());
    const auto n = eq_->arity();
    if (want_ != PyEquation::Want::Gradient) {
      f_.emplace(eq_->shape_of({outputs}, at_),
                 PyEquation::count(*l, &rt::OutputSpans::values), at_.size());
    }
    if (want_ != PyEquation::Want::Value) {
      partials_.emplace(PyEquation::count(*l, &rt::OutputSpans::jacobian),
                        at_.size());
      g_.emplace(eq_->shape_of({outputs, ssize(n)}, at_), at_.size());
    }
    if (want_ == PyEquation::Want::Hessian) {
      compressed_.emplace(PyEquation::count(*l, &rt::OutputSpans::hessian),
                          at_.size());
      h_.emplace(eq_->shape_of({outputs, ssize(n), ssize(n)}, at_), at_.size());
    }
  }

  // The lane is looked up per call, never cached: adopt() swaps a kernel in,
  // and a call holding the old one interprets on with every answer still
  // right.
  void operator()() {
    const auto l = eq_->lane(want_);
    eq_->run(
        want_, at_,
        {.values = f_ ? f_->rows() : std::span<double *const>{},
         .jacobian = partials_ ? partials_->rows() : std::span<double *const>{},
         .hessian =
             compressed_ ? compressed_->rows() : std::span<double *const>{}});
    if (g_) {
      eq_->scatter_jacobian(*partials_, *g_, at_.size());
    }
    if (h_) {
      scatter(*l);
    }
  }

  [[nodiscard]] pyb::object x() const { return x_; }
  [[nodiscard]] pyb::object value() const { return read(f_); }
  [[nodiscard]] pyb::object jacobian() const { return read(g_); }
  [[nodiscard]] pyb::object hessian() const { return read(h_); }

  [[nodiscard]] std::string repr() const {
    return std::format("<ddx.Call {} at {} point{}>",
                       boost::algorithm::to_lower_copy(
                           std::string{rt::name_of(want_)}),
                       at_.size(), at_.size() == 1 ? "" : "s");
  }

private:
  void scatter(const PyEquation::Compiled &l) {
    scatter_hessian(l.graph->coloring(), compressed_->rows(), h_->at(0),
                    eq_->arity(), at_.size());
  }

  // A rank-0 block is the scalar case, where the allocating calls answer with
  // a float; matching them costs a PyFloat only when the value is read.
  template <typename B>
  [[nodiscard]] static pyb::object read(const std::optional<B> &b) {
    if (!b) {
      fail_with(errc::wrong_column_count);
    }
    const Array &a = b->bound();
    return a.ndim() == 0 ? pyb::object{pyb::float_(*a.data())} : pyb::object{a};
  }

  PyEquation *eq_; // outlived by the equation: keep_alive on the factory
  PyEquation::Want want_;
  Array x_; // declared before at_, which points into it
  Point at_;
  std::optional<Block> f_;
  std::optional<Scratch> partials_; // the pattern's cells, on their way to g_
  std::optional<Dense> g_;
  std::optional<Scratch> compressed_; // the colouring's, on their way to h_
  std::optional<Dense> h_;
};

} // namespace ddx::py
