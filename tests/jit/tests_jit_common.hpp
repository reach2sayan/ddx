#pragma once
// Shared by every tests_jit_*.cpp: the one compiler, and a compile helper that
// reports LLVM's reason instead of returning an empty Kernel.

#include "jit/kernel.hpp"

#include <gtest/gtest.h>

#include <string>
#include <utility>

namespace {

// Without a native target there is no JIT to test, so report the reason once.
ddx::jit::Compiler &compiler() {
  static ddx::jit::compile_result<ddx::jit::Compiler> c =
      ddx::jit::Compiler::create();
  EXPECT_TRUE(c.has_value()) << (c ? "" : c.error().detail);
  return *c;
}

auto must_compile(auto &&...args) {
  auto k = compiler().compile(static_cast<decltype(args) &&>(args)...);
  EXPECT_TRUE(k.has_value()) << (k ? std::string{} : k.error().detail);
  return k ? std::move(*k) : typename decltype(k)::value_type{};
}

} // namespace
