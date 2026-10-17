"""The compiled path, and what a build without it does instead.

Both python presets run this file. Nothing is skipped by name: ``ddx.has_jit``
says whether a kernel can ever land, and the assertions branch on that -- so the
no-JIT build is *tested*, not merely excused.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import ddx

# One row per symbol, one column per point -- which is the kernel's own layout,
# so a batch shaped this way reaches it without a copy.
Batch = NDArray[np.float64]


@pytest.fixture
def batch() -> Batch:
    rng = np.random.default_rng(20260825)
    return np.stack([rng.uniform(0.2, 2.0, 64), rng.uniform(0.2, 2.0, 64)])


def test_interpret_is_the_default(scalar: ddx.Equation) -> None:
    assert scalar.options.backend is ddx.Backend.INTERPRET
    assert scalar.uses_kernel is False


def test_compile_lands_a_kernel_where_there_is_a_backend(
    scalar: ddx.Equation, batch: Batch
) -> None:
    scalar.compile(points=batch.shape[1])
    assert scalar.wait_for_kernel() is ddx.has_jit
    assert scalar.uses_kernel is ddx.has_jit


def test_adapt_compiles_nothing_until_the_batch_pays_for_it(
    scalar: ddx.Equation, batch: Batch
) -> None:
    scalar.options = ddx.Options(
        backend=ddx.Backend.ADAPT, points=batch.shape[1], warm_points=1 << 40
    )
    scalar.jacobian(batch)
    assert scalar.uses_kernel is False


def test_adapt_compiles_once_the_batch_pays_for_it(
    scalar: ddx.Equation, batch: Batch
) -> None:
    scalar.options = ddx.Options(
        backend=ddx.Backend.ADAPT, points=batch.shape[1], warm_points=1
    )
    swept_value, swept_jacobian = scalar.jacobian(batch)  # buys the rung
    assert scalar.wait_for_kernel() is ddx.has_jit
    assert scalar.uses_kernel is ddx.has_jit

    value, jacobian = scalar.jacobian(batch)
    assert np.array_equal(value, swept_value)
    assert np.array_equal(jacobian, swept_jacobian)


def test_the_kernel_and_the_sweep_agree(scalar: ddx.Equation, batch: Batch) -> None:
    """The kernel and the sweep agree bit for bit, not merely closely.

    `contract` is read by the *graph*, so either both paths fold a product into
    one rounding or neither does.
    """
    swept_value, swept_jacobian = scalar.jacobian(batch)

    scalar.compile(points=batch.shape[1])
    value, jacobian = scalar.jacobian(batch)

    assert np.array_equal(value, swept_value)
    assert np.array_equal(jacobian, swept_jacobian)


def test_a_compiled_hessian_agrees_too(scalar: ddx.Equation, batch: Batch) -> None:
    swept = scalar.hessian(batch)
    scalar.compile(points=batch.shape[1])
    compiled = scalar.hessian(batch)
    for a, b in zip(swept, compiled, strict=True):
        assert np.array_equal(a, b)


def test_answers_do_not_depend_on_the_backend(
    scalar: ddx.Equation, batch: Batch
) -> None:
    """A refused or absent compile leaves the sweep, and the sweep is right."""
    scalar.compile(points=batch.shape[1])
    compiled_value, _ = scalar.jacobian(batch)

    scalar.options = ddx.Options(backend=ddx.Backend.INTERPRET)
    assert scalar.uses_kernel is False
    swept_value, _ = scalar.jacobian(batch)
    assert np.array_equal(compiled_value, swept_value)


def test_a_system_compiles_too(system: ddx.Equation, batch: Batch) -> None:
    swept = system.jacobian(batch)
    system.compile(points=batch.shape[1])
    assert system.uses_kernel is ddx.has_jit
    compiled = system.jacobian(batch)
    for a, b in zip(swept, compiled, strict=True):
        assert np.array_equal(a, b)


def test_a_scalar_kernel_answers_a_single_point(scalar: ddx.Equation) -> None:
    """points=1 is the minimisation case: a gradient per step, at one point."""
    swept = scalar.jacobian([2.0, 3.0])
    scalar.compile(points=1)
    compiled = scalar.jacobian([2.0, 3.0])
    assert np.array_equal(swept[1], compiled[1])


def test_hessian_colours_are_read_without_compiling(scalar: ddx.Equation) -> None:
    assert scalar.hessian_colours == 2
    assert scalar.uses_kernel is False
