#pragma once
// The whole public compile-time surface; operators and math functions are
// found by ADL. Citations are in REFERENCES.md and at each "Ref:" line.
#include "symbolic/equation.hpp"
#include "util/version.hpp"

namespace ddx {

using impl::Equation;

// A double-valued Equation answers all but hessian(), which needs dual.
using impl::dual;
using impl::dual2nd;

// A forward dual N levels deep. dual_seed<N>(v, mask) makes a variable and
// dual_part<Mask>(f) reads one of f's 2^N coefficients.
template <typename T, std::size_t N> using nth_dual = impl::nth_dual_t<T, N>;
using impl::dual_part;
using impl::dual_seed;

// var_of<"x">(v) and dual_var_of<"x">(v) take their scalar type from v.
using impl::dual_var_of;
using impl::sym;
using impl::var;
using impl::var_of;
using impl::variable;
namespace literals = impl::literals;

using impl::constant;

// eq[idx<1>()] is the same as eq.get<1>().
using impl::idx;

// named<"x">(3.0) is one named entry of a point.
using impl::Entry;
using impl::named;

using impl::DiffMode;

} // namespace ddx
