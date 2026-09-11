// The one TU that talks to OpenCL
#include "cl/device.hpp"
#include "cl/kernel.hpp"
#include "codegen.hpp"

#include "rt/graph.hpp"
#include "util/ranges.hpp"

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/exception/program_build_failure.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <exception>
#include <format>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bc = boost::compute;

namespace ddx::cl {

struct Device::Impl {
  bc::device device;
  bc::context context;
  std::string identity;
};

// Everything one kernel owns.  The context is held so the buffers outlive the
// device's last handle; the buffers grow and never shrink.
struct Kernel::Impl {
  Impl(bc::context c, bc::command_queue q, bc::kernel k, std::size_t local)
      : context{std::move(c)}, queue{std::move(q)}, kernel{std::move(k)},
        local{local} {}

  bc::context context;
  bc::command_queue queue; // in order: writes, launch, reads
  bc::kernel kernel;
  std::size_t local;
  std::mutex mutex;
  std::array<bc::buffer, 4> buffers{};    // xs, f, g, h
  std::array<std::size_t, 4> capacity{}; // in doubles
};

namespace {

[[nodiscard]] std::unexpected<error> refused(errc code, std::string detail) {
  return std::unexpected{error{code, std::move(detail)}};
}

// The pragma needs the extension, and a device that lists it but reports no
// double configuration cannot honour the arithmetic.
[[nodiscard]] bool doubles(const bc::device &d) {
  return d.supports_extension("cl_khr_fp64") &&
         d.get_info<cl_device_fp_config>(CL_DEVICE_DOUBLE_FP_CONFIG) != 0;
}

struct Candidate {
  bc::device device;
  std::string name; // "<platform> / <device>", what a selector matches
  bool fp64;
};

[[nodiscard]] std::vector<Candidate> candidates() {
  std::vector<Candidate> out;
  for (const bc::platform &platform : bc::system::platforms()) {
    for (const bc::device &device : platform.devices()) {
      out.push_back({.device = device,
                     .name = std::format(
                         "{} / {}", boost::algorithm::trim_copy(platform.name()),
                         boost::algorithm::trim_copy(device.name())),
                     .fp64 = doubles(device)});
    }
  }
  return out;
}

// A GPU first when nothing is named: a CPU runtime is there to be asked for.
[[nodiscard]] const Candidate *choose(const std::vector<Candidate> &found,
                                      std::string_view selector) {
  auto usable = found | std::views::filter(&Candidate::fp64);
  const auto pick = [&usable](auto &&wanted) -> const Candidate * {
    const auto it = std::ranges::find_if(usable, wanted);
    return it == usable.end() ? nullptr : &*it;
  };
  if (!selector.empty()) {
    return pick([selector](const Candidate &c) {
      return boost::algorithm::icontains(c.name, selector);
    });
  }
  if (const Candidate *gpu = pick([](const Candidate &c) {
        return (c.device.type() & CL_DEVICE_TYPE_GPU) != 0;
      })) {
    return gpu;
  }
  return pick([](const Candidate &) { return true; });
}

[[nodiscard]] std::string seen(const std::vector<Candidate> &found) {
  return boost::algorithm::join(
      found | std::views::transform([](const Candidate &c) {
        return c.fp64 ? c.name : c.name + " (no fp64)";
      }) | impl::to<std::vector<std::string>>(),
      "; ");
}

// Before the caller's columns go back to it: an asynchronous copy still in
// flight would race the sweep that answers instead.
void drain(bc::command_queue &queue) noexcept {
  try {
    queue.finish();
  } catch (...) {
  }
}

// Up to 256 work-items a group, in the device's preferred multiple.
[[nodiscard]] std::size_t group_size(const bc::kernel &k, const bc::device &d) {
  const auto most = std::min<std::size_t>(
      k.get_work_group_info<std::size_t>(d, CL_KERNEL_WORK_GROUP_SIZE), 256);
  const auto step = k.get_work_group_info<std::size_t>(
      d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
  return std::max<std::size_t>(step > 0 && most >= step ? most - most % step : most, 1);
}

// One worker: a driver's compiler is already threaded, and the lanes of one
// equation queue behind each other as the LLVM ladder's do.  Constructed on
// the first compile, after the vendor runtime it calls into, so it is joined
// before that runtime tears down.
[[nodiscard]] boost::asio::thread_pool &compiles() {
  static boost::asio::thread_pool threads{1};
  return threads;
}

} // namespace

result<Device> Device::create(std::string_view selector) {
  try {
    const std::vector<Candidate> found = candidates();
    if (found.empty()) {
      return refused(errc::no_device, "no OpenCL platform is installed");
    }
    const Candidate *const chosen = choose(found, selector);
    if (chosen == nullptr) {
      return refused(errc::no_device,
                     selector.empty()
                         ? std::format("no device has double precision; saw: {}",
                                       seen(found))
                         : std::format("nothing with double precision matches "
                                       "'{}'; saw: {}",
                                       selector, seen(found)));
    }
    bc::context context{chosen->device};
    return Device{std::make_shared<Impl>(Impl{
        .device = chosen->device,
        .context = std::move(context),
        .identity = std::format("{} / {} / {}", chosen->name,
                                chosen->device.driver_version(),
                                chosen->device.version())})};
  } catch (const bc::opencl_error &e) {
    return refused(errc::no_device, e.error_string());
  } catch (const std::exception &e) {
    return refused(errc::no_device, e.what());
  }
}

const result<Device> &Device::shared(std::string_view selector) {
  struct Registry {
    std::mutex mutex;
    std::map<std::string, result<Device>, std::less<>> devices;
  };
  // Leaked on purpose: a context released while the program exits is what
  // vendor runtimes crash on, and nothing here needs releasing then.
  static Registry *const registry = new Registry;
  const std::scoped_lock lock{registry->mutex};
  auto it = registry->devices.find(selector);
  if (it == registry->devices.end()) {
    it = registry->devices.emplace(std::string{selector}, create(selector))
             .first;
  }
  return it->second;
}

std::string_view Device::identity() const noexcept { return impl_->identity; }

result<Kernel> Device::compile(const rt::Graph<double> &g,
                               CompileReport *report) const {
  using clock = std::chrono::steady_clock;
  CompileReport discard;
  CompileReport &rep = report != nullptr ? *report : discard;
  const auto start = clock::now();
  const std::string source = detail::emit_source(g);
  const auto emitted = clock::now();
  rep.nodes = g.schedule().size();
  rep.source_bytes = source.size();
  rep.emit = emitted - start;
  try {
    const bc::program program = bc::program::build_with_source(
        source, impl_->context, std::string{detail::build_options});
    bc::kernel kernel{program, std::string{detail::kernel_name}};
    const std::size_t local = group_size(kernel, impl_->device);
    rep.build = clock::now() - emitted;
    return Kernel{std::make_shared<Kernel::Impl>(
                      impl_->context,
                      bc::command_queue{impl_->context, impl_->device},
                      std::move(kernel), local),
                  jit::KernelShape::of(g)};
  } catch (const bc::program_build_failure &e) {
    return refused(errc::device_compile, e.build_log());
  } catch (const bc::opencl_error &e) {
    return refused(errc::device_compile, e.error_string());
  } catch (const std::exception &e) {
    return refused(errc::device_compile, e.what());
  }
}

// packaged_task, never std::async: only the latter's future joins in its
// destructor, putting a compile on the critical path of whoever dropped it.
std::shared_future<result<Kernel>>
Device::compile_async(std::shared_ptr<const rt::Graph<double>> g) const {
  auto task = std::make_shared<std::packaged_task<result<Kernel>()>>(
      [self = *this, graph = std::move(g)] { return self.compile(*graph); });
  auto landing = task->get_future().share();
  boost::asio::post(compiles(), [task] { std::invoke(*task); });
  return landing;
}

result<void> Kernel::operator()(std::span<const double *const> xs,
                                std::span<double *const> f,
                                std::span<double *const> g,
                                std::span<double *const> h,
                                std::size_t n) const {
  if (!impl_) {
    return refused(errc::device_launch, "an empty kernel");
  }
  // A mismatch past here is a read or write out of bounds on the device.
  assert(xs.size() == shape_.arity && f.size() == shape_.values &&
         g.size() == shape_.jacobian && h.size() == shape_.hessian);
  if (n == 0) {
    return {};
  }
  Impl &k = *impl_;
  const std::scoped_lock lock{k.mutex};
  const std::size_t bytes = n * sizeof(double);
  const std::array outputs{f, g, h};
  try {
    // Never a zero-sized buffer: OpenCL refuses one, and a block that was not
    // asked for still binds an argument.
    const auto sized = [&](std::size_t slot, std::size_t columns,
                           cl_mem_flags flags) -> const bc::buffer & {
      const std::size_t doubles = std::max<std::size_t>(columns, 1) * n;
      if (k.capacity[slot] < doubles) {
        k.buffers[slot] = bc::buffer{k.context, doubles * sizeof(double), flags};
        k.capacity[slot] = doubles;
      }
      return k.buffers[slot];
    };

    const bc::buffer &in = sized(0, xs.size(), CL_MEM_READ_ONLY);
    for (const auto [j, column] : std::views::enumerate(xs)) {
      k.queue.enqueue_write_buffer_async(in, static_cast<std::size_t>(j) * bytes,
                                         bytes, column);
    }
    k.kernel.set_arg(0, in);
    for (const auto [slot, columns] : std::views::enumerate(outputs)) {
      const auto arg = static_cast<std::size_t>(slot) + 1;
      k.kernel.set_arg(arg, sized(arg, columns.size(), CL_MEM_WRITE_ONLY));
    }
    k.kernel.set_arg(4, static_cast<cl_ulong>(n));
    const std::size_t global = (n + k.local - 1) / k.local * k.local;
    k.queue.enqueue_1d_range_kernel(k.kernel, 0, global, k.local);
    for (const auto [slot, columns] : outputs | std::views::enumerate) {
      const auto &from = k.buffers[static_cast<std::size_t>(slot) + 1];
      for (const auto [j, column] : columns | std::views::enumerate) {
        k.queue.enqueue_read_buffer_async(
            from, static_cast<std::size_t>(j) * bytes, bytes, column);
      }
    }
    k.queue.finish();
    return {};
  } catch (const bc::opencl_error &e) {
    drain(k.queue);
    return refused(errc::device_launch, e.error_string());
  } catch (const std::exception &e) {
    drain(k.queue);
    return refused(errc::device_launch, e.what());
  }
}

} // namespace ddx::cl
