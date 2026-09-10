"""The OpenCL device backend, and what a host without a device does instead.

Skipped as a whole on a build without the backend.  On one with it, a host with
no device still runs every test that needs none: a selector nothing matches is
the same answer as an empty machine.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import ddx

pytestmark = pytest.mark.skipif(not ddx.has_opencl, reason="built without the OpenCL backend")

Batch = NDArray[np.float64]


@pytest.fixture
def arithmetic() -> ddx.Equation:
    """Nothing but IEEE arithmetic, so the device has no room to differ."""

    @ddx.equation
    def f() -> ddx.Expression:
        x = ddx.var("x")
        y = ddx.var("y")
        return x * y + ddx.max(x, y) - ddx.select(x < y, x / y, y * y)

    return f


@pytest.fixture
def batch() -> Batch:
    rng = np.random.default_rng(20260910)
    return np.stack([rng.uniform(0.2, 2.0, 257), rng.uniform(0.2, 2.0, 257)])


def _on_device(eq: ddx.Equation, points: int) -> ddx.Equation:
    eq.compile(backend=ddx.Backend.DEVICE, points=points)
    try:
        eq.device_status  # noqa: B018  -- raising is the answer
    except ddx.Error as e:
        if e.code is ddx.errc.no_device:
            pytest.skip(str(e))
        raise
    return eq


def test_device_status_is_none_under_any_other_backend(scalar: ddx.Equation) -> None:
    assert scalar.device_status is None
    scalar.options = ddx.Options(backend=ddx.Backend.ADAPT)
    assert scalar.device_status is None


def test_the_device_and_the_sweep_agree_to_the_bit(
    arithmetic: ddx.Equation, batch: Batch
) -> None:
    swept_value, swept_jacobian = arithmetic.jacobian(batch)
    swept_hessian = arithmetic.hessian(batch)

    _on_device(arithmetic, batch.shape[1])
    assert arithmetic.uses_kernel
    assert arithmetic.wait_for_kernel(want=ddx.Want.HESSIAN)
    value, jacobian = arithmetic.jacobian(batch)
    hessian = arithmetic.hessian(batch)

    assert np.array_equal(value, swept_value)
    assert np.array_equal(jacobian, swept_jacobian)
    for got, want in zip(hessian, swept_hessian, strict=True):
        assert np.array_equal(got, want)


def test_a_model_with_transcendentals_agrees_closely(
    scalar: ddx.Equation, batch: Batch
) -> None:
    swept_value, swept_jacobian = scalar.jacobian(batch)
    _on_device(scalar, batch.shape[1])
    value, jacobian = scalar.jacobian(batch)
    np.testing.assert_allclose(value, swept_value, rtol=1e-12)
    np.testing.assert_allclose(jacobian, swept_jacobian, rtol=1e-10, atol=1e-12)


def test_a_selector_nothing_matches_sweeps_and_says_why(
    arithmetic: ddx.Equation, batch: Batch
) -> None:
    swept_value, swept_jacobian = arithmetic.jacobian(batch)
    arithmetic.options = ddx.Options(
        backend=ddx.Backend.DEVICE, device="no-such-device-xyzzy"
    )
    assert arithmetic.wait_for_kernel() is False
    assert arithmetic.uses_kernel is False
    with pytest.raises(ddx.Error) as raised:
        arithmetic.device_status  # noqa: B018  -- raising is the answer
    assert raised.value.code is ddx.errc.no_device

    value, jacobian = arithmetic.jacobian(batch)
    assert np.array_equal(value, swept_value)
    assert np.array_equal(jacobian, swept_jacobian)


def test_the_selector_is_an_option_and_round_trips() -> None:
    options = ddx.Options(backend=ddx.Backend.DEVICE, device="gfx")
    assert ddx.Options.model_validate_json(options.model_dump_json()) == options
    assert options._native().device == "gfx"
