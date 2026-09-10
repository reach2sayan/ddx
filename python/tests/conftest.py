"""Shared fixtures.

The models here are the ones ``tests/rt/`` already asserts numbers for, so the
two suites can be compared line for line.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

import ddx

# ctest points this at the package it just built.  An editable install registers
# a sys.meta_path finder, which beats every sys.path entry, so PYTHONPATH alone
# does not pin the module under test.  Failing silently: the suite passes against
# the wrong build, and the no-JIT preset is the case that matters.
_EXPECTED = os.environ.get("DDX_PACKAGE_DIR")
if _EXPECTED and Path(ddx.__file__).parent != Path(_EXPECTED).resolve() / "ddx":
    raise RuntimeError(
        f"imported ddx from {Path(ddx.__file__).parent}, not the build tree at "
        f"{_EXPECTED}.  An editable install shadows it; `uv pip uninstall ddx-ad` "
        f"or install non-editable."
    )


@dataclass(frozen=True, slots=True)
class Derivatives:
    """What ``exp(x) sin(y)`` and its derivatives come to at a point."""

    f: float
    dx: float
    dy: float
    dxx: float
    dxy: float
    dyy: float


@pytest.fixture
def scalar() -> ddx.Equation:
    """Return ``exp(x) * sin(y)``, as in ``tests_rt_equation.cpp``."""

    @ddx.equation
    def eq() -> ddx.Expression:
        """exp(x) * sin(y)."""
        x = ddx.var("x")
        y = ddx.var("y")
        return ddx.exp(x) * ddx.sin(y)

    return eq


@pytest.fixture
def system() -> ddx.Equation:
    """Return the Newton system from ``tests_rt_newton.cpp``."""

    @ddx.equation
    def eq() -> tuple[ddx.Expression, ddx.Expression]:
        """A circle and a hyperbola."""  # noqa: D401
        x = ddx.var("x")
        y = ddx.var("y")
        return x * x + y * y - 4.0, x * y - 1.0

    return eq


@pytest.fixture
def expected() -> Derivatives:
    """Return ``exp(x) sin(y)`` and its derivatives at (2, 3)."""
    e, s, c = math.exp(2.0), math.sin(3.0), math.cos(3.0)
    return Derivatives(f=e * s, dx=e * s, dy=e * c, dxx=e * s, dxy=e * c, dyy=-e * s)
