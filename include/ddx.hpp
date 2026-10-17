#pragma once
// The public surface, and the whole of it; everything else lives in ddx::impl.
// Operators and math functions are found by ADL, so they are not re-exported.
// REFERENCES.md carries the algorithms; per-derivation citations sit at the
// code that implements them -- grep "Ref:".
#include "symbolic/equation.hpp"
#include "util/version.hpp"

namespace ddx {

// The only derivative entry point.
using impl::Equation;

// A `double` graph answers everything but hessian(), which needs `dual`.
using impl::dual;
using impl::dual2nd;

// var_of<"x">(v) / dual_var_of<"x">(v) take the scalar from an exemplar.
using impl::dual_var_of;
using impl::sym;
using impl::var;
using impl::var_of;
using impl::variable;
namespace literals = impl::literals;

// constant(3.0) -- a value stored in the tree.
using impl::constant;

// eq[idx<1>()] -- the subscript spelling of Equation::get<N>().
using impl::idx;

// named<"x">(3.0) -- one keyword argument of a point, or one record entry.
using impl::Entry;
using impl::named;

// Symbolic evaluates the stored partial trees; Reverse builds none.
using impl::DiffMode;

} // namespace ddx
