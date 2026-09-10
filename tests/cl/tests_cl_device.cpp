#include "tests_cl_common.hpp"

#include "rt/derivative.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace {

using ddx::rt::Builder;
using ddx::rt::GraphBuilder;
using ClDevice = cltest::OnDevice;

// These three need no device: a host with none answers them the same way.
TEST(ClDeviceSelection, NothingMatchingIsNoDevice) {
  const auto d = ddx::cl::Device::create("no-such-device-xyzzy");
  ASSERT_FALSE(d);
  EXPECT_EQ(d.error().code, ddx::errc::no_device);
  EXPECT_FALSE(d.error().detail.empty());
}

TEST(ClDeviceSelection, SharedIsOnePerSelector) {
  EXPECT_EQ(&ddx::cl::Device::shared("no-such-device-xyzzy"),
            &ddx::cl::Device::shared("no-such-device-xyzzy"));
}

TEST(ClDeviceSelection, AnEmptyKernelIsFalsyAndRefusesToRun) {
  const ddx::cl::Kernel k;
  EXPECT_FALSE(k);
  const auto ran = k({}, {}, {}, {}, 1);
  ASSERT_FALSE(ran);
  EXPECT_EQ(ran.error().code, ddx::errc::device_launch);
}

// "<platform> / <device>" is what a selector matches, and the identity leads
// with it.
TEST_F(ClDevice, TheIdentityLeadsWithWhatASelectorMatches) {
  const std::string_view identity = cltest::device()->identity();
  const auto platform_end = identity.find(" / ");
  ASSERT_NE(platform_end, std::string_view::npos) << identity;
  const auto device_end = identity.find(" / ", platform_end + 3);
  ASSERT_NE(device_end, std::string_view::npos) << identity;
  const auto again = ddx::cl::Device::create(identity.substr(0, device_end));
  ASSERT_TRUE(again) << again.error().detail;
  EXPECT_EQ(again->identity(), identity);
}

TEST_F(ClDevice, TheShapeIsTheGraphsAndTheReportIsFilled) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto g =
      GraphBuilder{b}.value(x * y + exp(x)).build_jacobian().build_hessian().finish();
  ddx::cl::CompileReport report;
  const auto k = cltest::device()->compile(g, &report);
  ASSERT_TRUE(k) << k.error().detail;
  EXPECT_EQ(k->shape(), ddx::jit::KernelShape::of(g));
  EXPECT_EQ(report.nodes, g.schedule().size());
  EXPECT_GT(report.source_bytes, 0u);
}

TEST_F(ClDevice, AnEmptyBatchWritesNothing) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto g = GraphBuilder{b}.value(x * x).build_jacobian().finish();
  const auto k = cltest::device()->compile(g);
  ASSERT_TRUE(k) << k.error().detail;
  double in = 2.0;
  double f = cltest::sentinel;
  double dx = cltest::sentinel;
  const double *const xs[] = {&in};
  double *const fs[] = {&f};
  double *const gs[] = {&dx};
  EXPECT_TRUE((*k)(xs, fs, gs, {}, 0));
  EXPECT_EQ(f, cltest::sentinel);
  EXPECT_EQ(dx, cltest::sentinel);
}

// The kernel holds the context its buffers live in, so it runs after the
// device handle that built it has gone.
TEST_F(ClDevice, AKernelOutlivesItsDevice) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto g = GraphBuilder{b}.value(x * x + 1.0).build_jacobian().finish();
  const auto k = [&g] {
    const auto d = ddx::cl::Device::create(cltest::selector());
    return d->compile(g);
  }();
  ASSERT_TRUE(k) << k.error().detail;
  const auto out = cltest::ran(*k, {{3.0, -2.0}}, 2);
  ASSERT_TRUE(out) << out.error().detail;
  EXPECT_EQ((*out)[0], (std::vector{10.0, 5.0}));
  EXPECT_EQ((*out)[1], (std::vector{6.0, -4.0}));
}

} // namespace
