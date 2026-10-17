#pragma once
// Shared by every tests_jit_*.cpp: the one compiler, and a compile that names
// LLVM's reason rather than meeting an empty Kernel three lines later.

#include "jit/kernel.hpp"

#include <gtest/gtest.h>

#include <string>
#include <utility>

namespace {

// No native target means no JIT to test; bring it up once and name the reason.
ddx::jit::Compiler &compiler() {
  static ddx::jit::compile_result<ddx::jit::Compiler> c =
      ddx::jit::Compiler::create();
  EXPECT_TRUE(c.has_value()) << (c ? "" : c.error().detail);
  return *c;
}

ddx::jit::Kernel must_compile(auto &&...args) {
  auto k = compiler().compile(static_cast<decltype(args) &&>(args)...);
  EXPECT_TRUE(k.has_value()) << (k ? std::string{} : k.error().detail);
  return k ? std::move(*k) : ddx::jit::Kernel{};
}

} // namespace
