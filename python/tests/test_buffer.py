"""The bound call: buffers taken once, then written into and read back.

``Equation.buffer`` is what an allocating ``jacobian(x)`` is not -- a call that
rebuilds no numpy array, no shape, no row-pointer vector and no tuple.  The
answers must be the allocating call's, to the bit, which is what these check.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import Derivatives

import ddx


def test_matches_the_allocating_call(
    scalar: ddx.Equation, expected: Derivatives
) -> None:
    call = scalar.buffer(np.array([2.0, 3.0]))
    call()

    assert call.value == pytest.approx(expected.f)
    assert call.jacobian == pytest.approx([expected.dx, expected.dy])


def test_a_gradient_call_fills_the_jacobian_and_nothing_else(
    scalar: ddx.Equation, expected: Derivatives
) -> None:
    call = scalar.buffer(np.array([2.0, 3.0]), want=ddx.Want.GRADIENT)
    call()
    assert call.jacobian == pytest.approx([expected.dx, expected.dy])
    with pytest.raises(ddx.Error):
        _ = call.value


def test_the_value_is_a_float_where_the_allocating_call_says_so(
    scalar: ddx.Equation,
) -> None:
    """One output at one point answers with a float, not a rank-0 array."""
    call = scalar.buffer(np.array([2.0, 3.0]))
    call()

    assert isinstance(call.value, float)


def test_the_point_moves_through_x(scalar: ddx.Equation) -> None:
    """The bound array *is* the point; writing into it is how the next one arrives."""
    call = scalar.buffer(np.array([2.0, 3.0]))
    call()
    call.x[:] = [0.5, 1.25]
    call()

    value, gradient = scalar.jacobian([0.5, 1.25])
    assert call.value == pytest.approx(value)
    assert call.jacobian == pytest.approx(gradient)


def test_the_blocks_are_the_same_arrays_every_call(scalar: ddx.Equation) -> None:
    """Nothing is allocated per call, which is the whole point of binding."""
    call = scalar.buffer(np.array([2.0, 3.0]))
    call()
    first = call.jacobian
    call()

    assert call.jacobian is first


def test_a_batch_binds_too(scalar: ddx.Equation) -> None:
    points = np.array([[2.0, 0.5, 1.0], [3.0, 1.25, 2.0]])
    call = scalar.buffer(points.copy())
    call()

    values, gradients = scalar.jacobian(points)
    assert call.value == pytest.approx(values)
    assert call.jacobian == pytest.approx(gradients)


def test_a_values_call_computes_no_gradient(scalar: ddx.Equation) -> None:
    """A block nobody asked for is work nobody does, and there is none to read."""
    call = scalar.buffer(np.array([2.0, 3.0]), want=ddx.Want.VALUE)
    call()

    assert call.value == pytest.approx(scalar.evaluate([2.0, 3.0]))
    with pytest.raises(ddx.Error) as raised:
        _ = call.jacobian
    assert raised.value.code == ddx.errc.wrong_column_count


def test_a_hessian_call_fills_all_three(
    scalar: ddx.Equation, expected: Derivatives
) -> None:
    call = scalar.buffer(np.array([2.0, 3.0]), want=ddx.Want.HESSIAN)
    call()

    assert call.value == pytest.approx(expected.f)
    assert call.jacobian == pytest.approx([expected.dx, expected.dy])
    assert call.hessian == pytest.approx(
        np.array([[expected.dxx, expected.dxy], [expected.dxy, expected.dyy]])
    )


def test_a_system_binds_its_jacobian(system: ddx.Equation) -> None:
    call = system.buffer(np.array([1.5, 1.0]))
    call()

    values, jacobian = system.jacobian([1.5, 1.0])
    assert call.value == pytest.approx(values)
    assert call.jacobian == pytest.approx(jacobian)


def test_a_system_has_no_hessian_lane_to_bind(system: ddx.Equation) -> None:
    """A frozen graph carries one colouring, so m of them are read off the arena.

    The refusal comes from `ready(want)`, the single gate every point and batch
    call passes: the Want row says a Hessian needs one output, so a system is
    turned away as `not_single_output` and says so.
    """
    with pytest.raises(ddx.Error) as raised:
        system.buffer(np.array([1.5, 1.0]), want=ddx.Want.HESSIAN)
    assert raised.value.code == ddx.errc.not_single_output


def test_a_list_point_is_bound_as_an_array(scalar: ddx.Equation) -> None:
    """Whatever came in, `x` is the array the call reads -- so it can be written."""
    call = scalar.buffer([2.0, 3.0])
    call()
    call.x[:] = [0.5, 1.25]
    call()

    assert call.value == pytest.approx(scalar.evaluate([0.5, 1.25]))


@pytest.mark.skipif(not ddx.has_jit, reason="needs the LLVM backend")
def test_a_kernel_landing_mid_loop_is_picked_up(scalar: ddx.Equation) -> None:
    """The lane is looked up per call, so a compile that lands is adopted.

    Caching the kernel would leave a bound call interpreting forever with every
    answer still right, which is why this asserts the kernel and the numbers.
    """
    call = scalar.buffer(np.array([2.0, 3.0]))
    call()
    scalar.compile(points=1)
    for _ in range(16):
        call()

    assert scalar.uses_kernel
    value, gradient = scalar.jacobian([2.0, 3.0])
    assert call.value == pytest.approx(value)
    assert call.jacobian == pytest.approx(gradient)


@pytest.mark.parametrize(
    "want", [ddx.Want.HVP, ddx.Want.JVP, ddx.Want.VJP, ddx.Want.DENSE_HESSIAN]
)
def test_a_want_the_buffer_cannot_bind_is_refused_up_front(
    scalar: ddx.Equation, want: ddx.Want
) -> None:
    with pytest.raises(ddx.Error) as caught:
        scalar.buffer(np.array([2.0, 3.0]), want=want)
    assert caught.value.code is ddx.errc.wrong_column_count
