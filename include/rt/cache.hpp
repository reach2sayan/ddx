#pragma once

#include "rt/cone.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "util/config.hpp" // DDX_FWD

#include <boost/container/small_vector.hpp> // a vector<bool> that holds bools
#include <boost/dynamic_bitset.hpp>

#include <algorithm>
#include <concepts>
#include <cstring> // std::memcmp, the only key comparison there is
#include <functional>
#include <memory>
#include <mutex> // unique_lock and try_to_lock, which <shared_mutex> lacks
#include <ranges>
#include <shared_mutex>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// The last call per lane, kept.  NOTES.md, "Remembering the last call".
namespace ddx::rt {

// A kernel computes its whole graph, so a lane it answers leaves no tape.
enum class Answered : std::uint8_t { BySweep, ByKernel };

// One remembered call: the input columns as passed, every output column end to
// end, and the node values the sweep left -- empty where a kernel answered.
template <typename T> struct Recorded {
  std::span<T> point;
  std::span<T> values;
  std::span<T> tape;
};

// What a slot holds and what it holds it for.  Two calls share an entry only
// where their whole Extent matches, so a re-freeze, a grown arena or a kernel
// arriving over a sweep all part one from the other.
class Extent {
public:
  [[nodiscard]] friend constexpr bool operator==(const Extent &,
                                                 const Extent &) = default;

  [[nodiscard]] constexpr std::size_t point() const noexcept { return point_; }
  [[nodiscard]] constexpr std::size_t values() const noexcept {
    return values_;
  }
  [[nodiscard]] constexpr std::size_t tape() const noexcept { return tape_; }

  // `nodes` is the arena's size, not the graph's: a borrowed arena may have
  // grown since the freeze, and live ids index below that.
  template <impl::Numeric T>
  [[nodiscard]] static Extent of(const Graph<T> &graph, Answered how,
                                 std::size_t nodes) {
    const Layout &layout = graph.layout();
    return Extent{graph.arity(),
                  layout.values + layout.jacobian + layout.hessian,
                  how == Answered::ByKernel ? 0uz : nodes, &graph};
  }

private:
  constexpr Extent(std::size_t point, std::size_t values, std::size_t tape,
                   const void *of) noexcept
      : point_(point), values_(values), tape_(tape), of_(of) {}

  std::size_t point_ = 0;
  std::size_t values_ = 0;
  std::size_t tape_ = 0;
  const void *of_ = nullptr; // which frozen graph, by identity

public:
  constexpr Extent() noexcept = default;
};

// A lease on one lane's slot, held while it is read or written.  Disengaged
// where the cache declines, or where another thread is writing.
template <typename L, typename T>
concept CReadLease = requires(const L &l) {
  { static_cast<bool>(l) } -> std::same_as<bool>;
  { l.filled() } -> std::same_as<bool>; // holds a previous call
  { l.entry() } -> std::same_as<Recorded<const T>>;
};

template <typename L, typename T>
concept CWriteLease = requires(L &l, const L &cl) {
  { static_cast<bool>(cl) } -> std::same_as<bool>;
  { cl.filled() } -> std::same_as<bool>;
  { l.entry() } -> std::same_as<Recorded<T>>;
  { l.commit() } -> std::same_as<void>; // what was written is the entry now
};

// Somewhere to put one remembered call per Want, and nothing else: the equation
// decides what is stale.  Both spellings are const, so a cache carries its own
// lock; the object is copied in and never referred to again.
template <typename C, typename T>
concept CValueCache = std::copy_constructible<C> && std::movable<C> &&
                      requires(const C &c, Want w, Extent e) {
                        // Whether any call could hit: false skips the gather.
                        { c.active() } -> std::same_as<bool>;
                        { c.read(w, e) } -> CReadLease<T>;
                        { c.write(w, e) } -> CWriteLease<T>;
                        { c.clear() }; // a re-freeze invalidates every lane
                      };

// One slot per lane behind one shared_mutex per lane.  The state sits behind a
// pointer so an Equation holding one is still movable, and a copy is a copy of
// the configuration: entries belong to the object that filled them.
template <impl::Numeric T = double> class LastValue {
public:
  LastValue() : lanes_(std::make_unique<Lanes>()) {}
  explicit LastValue(bool on)
      : lanes_{on ? std::make_unique<Lanes>() : nullptr} {}

  LastValue(const LastValue &other)
      : lanes_{other.lanes_ ? std::make_unique<Lanes>() : nullptr} {}
  LastValue &operator=(const LastValue &other) {
    lanes_ = other.lanes_ ? std::make_unique<Lanes>() : nullptr;
    return *this;
  }
  LastValue(LastValue &&) noexcept = default;
  LastValue &operator=(LastValue &&) noexcept = default;
  ~LastValue() = default;

private:
  struct Slot {
    mutable std::shared_mutex mutex;
    Extent extent{};
    std::vector<T> point, values, tape;
    bool filled = false;
  };
  using Lanes = std::array<Slot, want_count>;

public:
  class ReadLease {
  public:
    ReadLease() = default;
    ReadLease(const Slot &slot, std::shared_lock<std::shared_mutex> lock)
        : slot_(&slot), lock_(std::move(lock)) {}

    [[nodiscard]] explicit operator bool() const noexcept {
      return slot_ != nullptr;
    }
    [[nodiscard]] bool filled() const noexcept { return slot_->filled; }
    [[nodiscard]] Recorded<const T> entry() const noexcept {
      return {slot_->point, slot_->values, slot_->tape};
    }

  private:
    const Slot *slot_ = nullptr;
    std::shared_lock<std::shared_mutex> lock_;
  };

  class WriteLease {
  public:
    WriteLease() = default;
    WriteLease(Slot &slot, std::unique_lock<std::shared_mutex> lock)
        : slot_(&slot), lock_(std::move(lock)) {}

    [[nodiscard]] explicit operator bool() const noexcept {
      return slot_ != nullptr;
    }
    [[nodiscard]] bool filled() const noexcept { return slot_->filled; }
    [[nodiscard]] Recorded<T> entry() noexcept {
      return {slot_->point, slot_->values, slot_->tape};
    }
    void commit() noexcept { slot_->filled = true; }

  private:
    Slot *slot_ = nullptr;
    std::unique_lock<std::shared_mutex> lock_;
  };

  [[nodiscard]] bool active() const noexcept { return lanes_ != nullptr; }

  // Engaged only where the slot is already the shape asked for: resizing is a
  // write, and a reader holds nothing but a shared lock.
  [[nodiscard]] ReadLease read(Want want, const Extent &extent) const {
    if (!lanes_) {
      return {};
    }
    const Slot &slot = (*lanes_)[std::to_underlying(want)];
    std::shared_lock lock{slot.mutex};
    return slot.extent == extent ? ReadLease{slot, std::move(lock)}
                                 : ReadLease{};
  }

  // Never waits: waiting is the one outcome that would make a cached call
  // slower than an uncached one.
  [[nodiscard]] WriteLease write(Want want, const Extent &extent) const {
    if (!lanes_) {
      return {};
    }
    Slot &slot = (*lanes_)[std::to_underlying(want)];
    std::unique_lock lock{slot.mutex, std::try_to_lock};
    if (!lock) {
      return {};
    }
    if (slot.extent != extent) {
      slot.extent = extent;
      slot.point.assign(extent.point(), T{});
      slot.values.assign(extent.values(), T{});
      slot.tape.assign(extent.tape(), T{});
      slot.filled = false;
    }
    return {slot, std::move(lock)};
  }

  void clear() const {
    if (!lanes_) {
      return;
    }
    for (Slot &slot : *lanes_) {
      const std::unique_lock lock{slot.mutex};
      slot.extent = {};
      slot.filled = false;
    }
  }

private:
  mutable std::unique_ptr<Lanes> lanes_;
};

// The contract above is the whole of what a cache of one's own has to model;
// what follows is ours.
namespace detail {

// Empty, and Caching skips it outright rather than trusting the optimiser to.
template <impl::Numeric T> struct NoCache {
  struct Lease {
    [[nodiscard]] explicit operator bool() const noexcept { return false; }
    [[nodiscard]] bool filled() const noexcept { return false; }
    [[nodiscard]] Recorded<const T> entry() const noexcept { return {}; }
    [[nodiscard]] Recorded<T> entry() noexcept { return {}; }
    void commit() noexcept {}
  };
  [[nodiscard]] static constexpr bool active() noexcept { return false; }
  [[nodiscard]] constexpr Lease read(Want, const Extent &) const noexcept {
    return {};
  }
  [[nodiscard]] constexpr Lease write(Want, const Extent &) const noexcept {
    return {};
  }
  constexpr void clear() const noexcept {}
};

// Every single-point call in both equations comes through here.  Derived
// supplies arena() and unlocked() -- the GIL, on the Python side.
template <typename Derived, impl::Numeric T, CValueCache<T> Cache>
class Caching {
protected:
  constexpr Caching() = default;
  constexpr void take_cache(Cache c) { cache_ = std::move(c); }
  [[nodiscard]] constexpr const Cache &cache() const noexcept { return cache_; }

  // `run` is what the call would have done with no cache at all.
  void through(Want want, const Graph<T> &graph, Answered how,
               std::span<const T *const> xs, const Columns<T> &out,
               std::size_t n, auto &&run) const {
    if constexpr (std::same_as<Cache, NoCache<T>>) {
      run();
    } else {
      // A batch is amortised already, and comparing points would cost what it
      // saves; a cache built disabled would gather the point for nothing.
      if (n != 1 || !cache().active()) {
        run();
      } else {
        serve(want, graph, how, xs, out, DDX_FWD(run));
      }
    }
  }

private:
  [[nodiscard]] const Derived &self() const noexcept {
    return static_cast<const Derived &>(*this);
  }

  // What an entry is worth to the point in hand, as four types rather than a
  // ladder of bools; which sweep runs is then an overload.
  struct Served { // asked before; the answer is to hand
    std::span<const T> values;
  };
  struct Fill {}; // nothing to work from: every step
  struct Amend {  // only the cone the move reached
    std::span<const bool> changed;
  };
  struct Relay {}; // a kernel answers this lane: all or none
  using Plan = std::variant<Served, Fill, Amend, Relay>;

  void serve(Want want, const Graph<T> &graph, Answered how,
             std::span<const T *const> xs, const Columns<T> &out,
             auto &&run) const {
    const Extent extent = Extent::of(graph, how, self().arena().size());

    // The point, once, in the order a sweep reads it.
    std::vector<T> &at = point_scratch();
    at.resize(xs.size());
    std::ranges::transform(xs, at.begin(),
                           [](const T *column) { return *column; });

    if (const auto lease = cache().read(want, extent);
        lease && lease.filled() && unmoved(at, lease.entry().point)) {
      spread(lease.entry().values, out);
      return;
    }

    auto lease = cache().write(want, extent);
    if (!lease) { // another thread is filling this lane; do without
      run();
      return;
    }
    const Recorded<T> slot = lease.entry();
    std::visit(
        [&]<typename P>(const P &plan) {
          if constexpr (std::same_as<P, Served>) {
            spread(plan.values, out);
            return;
          } else if constexpr (std::same_as<P, Relay>) {
            run();
            collect(out, slot.values);
          } else { // Fill or Amend: whichever sweep the overload names
            sweep(want, graph, at, plan, slot);
            keep(graph.output_blocks(), slot.tape, slot.values);
            spread(slot.values, out);
          }
          remember(at, slot, lease);
        },
        decide(lease.filled(), at, slot));
  }

  [[nodiscard]] Plan decide(bool filled, const std::vector<T> &at,
                            const Recorded<T> &slot) const {
    // A tape is what an amendment amends, and a kernel leaves none.
    const bool swept = !slot.tape.empty();
    if (!filled) {
      return swept ? Plan{Fill{}} : Plan{Relay{}};
    }
    auto &changed = column_scratch();
    changed.assign(at.size(), false);
    std::ranges::transform(
        at, slot.point, changed.begin(),
        [](const T &now, const T &was) { return !same_bits(now, was); });
    // The thread that won the upgrade may have filled the entry with this very
    // point while this one waited for it.
    if (std::ranges::none_of(changed, std::identity{})) {
      return Served{slot.values};
    }
    return swept ? Plan{Amend{{changed.data(), changed.size()}}}
                 : Plan{Relay{}};
  }

  // Not an amendment with everything moved: a literal is dirty for no point at
  // all, and its slot still has to be written once.
  void sweep(Want, const Graph<T> &graph, const std::vector<T> &at, Fill,
             const Recorded<T> &slot) const {
    [[maybe_unused]] const auto held = self().unlocked();
    evaluate_into(self().arena(), at, graph.schedule(), slot.tape);
  }

  void sweep(Want want, const Graph<T> &graph, const std::vector<T> &at,
             const Amend &plan, const Recorded<T> &slot) const {
    const Cone &cone = cone_for(want, graph);
    // Given up around the sweep and nothing else: the GIL, on the Python side.
    [[maybe_unused]] const auto held = self().unlocked();
    evaluate_reached(self().arena(), at, graph.schedule(), cone, plan.changed,
                     reach_scratch(), stack_scratch(), slot.tape);
  }

  // Read the first time an amendment wants it, and only ever while this lane's
  // write lease is held -- which is what the mutable member below is safe under.
  [[nodiscard]] const Cone &cone_for(Want want, const Graph<T> &graph) const {
    Reached &kept = cones_[std::to_underlying(want)];
    if (kept.of != &graph) {
      kept = Reached{&graph,
                     Cone{self().arena(), graph.schedule(), graph.arity()}};
    }
    return kept.cone;
  }

  static void remember(const std::vector<T> &at, const Recorded<T> &slot,
                       auto &lease) {
    std::ranges::copy(at, slot.point.begin());
    lease.commit();
  }

  // Bitwise, never ==: -0.0 is not 0.0 as a point -- ask 1/x -- and a repeated
  // NaN is a hit, which == would refuse forever.
  [[nodiscard]] static bool same_bits(const T &a, const T &b) noexcept {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
  }

  [[nodiscard]] static bool unmoved(const std::vector<T> &at,
                                    std::span<const T> was) noexcept {
    return std::ranges::equal(at, was, same_bits);
  }

  // A remembered call is one run of values, block after block.
  static constexpr void spread(std::span<const T> values, const Columns<T> &out) {
    for (const auto [column, value] :
         std::views::zip(in_order(out) | std::views::join, values)) {
      *column = value;
    }
  }

  static constexpr void collect(const Columns<T> &out, std::span<T> values) {
    std::ranges::transform(in_order(out) | std::views::join, values.begin(),
                           [](const T *column) { return *column; });
  }

  static constexpr void keep(const OutputSpans &blocks, std::span<const T> tape,
                   std::span<T> values) {
    std::ranges::transform(in_order(blocks) | std::views::join, values.begin(),
                           [tape](NodeId o) { return tape[o]; });
  }

  // thread_local rather than members: a const call may come from two threads at
  // once and none of this is remembered.
  [[nodiscard]] static std::vector<T> &point_scratch() {
    static thread_local std::vector<T> at;
    return at;
  }
  [[nodiscard]] static boost::container::small_vector<bool, 64> &
  column_scratch() {
    // A Boost vector because std::vector<bool> is a bitset and cannot lend a
    // span; small, so a typical arity never allocates.
    static thread_local boost::container::small_vector<bool, 64> changed;
    return changed;
  }
  [[nodiscard]] static boost::dynamic_bitset<> &reach_scratch() {
    static thread_local boost::dynamic_bitset<> steps;
    return steps;
  }
  [[nodiscard]] static Cone::Stack &stack_scratch() {
    static thread_local Cone::Stack pending;
    return pending;
  }

  struct Reached {
    const void *of = nullptr; // which frozen graph it was read from
    Cone cone;
  };
  // Nothing at all without a cache: a constant evaluation can hold no graph.
  using Cones =
      std::conditional_t<std::same_as<Cache, NoCache<T>>, std::tuple<>,
                         std::array<Reached, want_count>>;

  [[no_unique_address]] Cache cache_{};
  [[no_unique_address]] mutable Cones cones_{};
};

} // namespace detail

} // namespace ddx::rt
