#pragma once
// How Equation::jacobian differentiates. Reverse, the default, yields every
// partial in one sweep; Symbolic evaluates the stored partial trees.
// Ref: Baur & Strassen, Theoret. Comput. Sci. 22(3) (1983) 317.

namespace ddx::impl {

enum class DiffMode { Symbolic, Reverse };

} // namespace ddx::impl
