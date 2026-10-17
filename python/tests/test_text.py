"""A model written down instead of built.

The property under all of it is that text is a second spelling and not a second
implementation: the same model reaches the same answers, whichever way it was
said.  The fixtures it compares against are the ones ``tests/rt/`` asserts
numbers for.
"""

from __future__ import annotations

import pytest
from conftest import Derivatives

import ddx


def test_matches_the_model_built_by_hand(
    scalar: ddx.Equation, expected: Derivatives
) -> None:
    written = ddx.equation("exp(x) * sin(y)")

    assert written.symbols == scalar.symbols
    assert written.evaluate([2.0, 3.0]) == pytest.approx(expected.f)

    value, gradient = written.jacobian([2.0, 3.0])
    assert value == pytest.approx(expected.f)
    assert gradient == pytest.approx([expected.dx, expected.dy])


def test_a_list_of_strings_is_a_system(system: ddx.Equation) -> None:
    written = ddx.equation(["x*x + y*y - 4", "x*y - 1"])

    assert written.symbols == system.symbols
    got_value, got_jacobian = written.jacobian([1.7, 0.6])
    want_value, want_jacobian = system.jacobian([1.7, 0.6])
    assert got_value == pytest.approx(want_value)
    assert got_jacobian == pytest.approx(want_jacobian)


def test_a_string_does_not_reach_the_model_overload() -> None:
    """A str is an object too, so this pins which overload claims it."""
    assert ddx.equation("x * 2").evaluate([3.0]) == pytest.approx(6.0)


def test_the_decorator_still_takes_a_model(scalar: ddx.Equation) -> None:
    """The text overloads are registered first; this is the regression on it."""

    @ddx.equation
    def eq() -> ddx.Expression:
        """exp(x) * sin(y)."""
        x = ddx.var("x")
        y = ddx.var("y")
        return ddx.exp(x) * ddx.sin(y)

    assert eq.__name__ == "eq"
    assert eq.evaluate([2.0, 3.0]) == pytest.approx(scalar.evaluate([2.0, 3.0]))


def test_python_spellings_and_python_readings() -> None:
    # `**` is the power, right-associative, and `^` is not exponentiation.
    assert ddx.equation("x ** y ** z").evaluate([2.0, 3.0, 2.0]) == pytest.approx(512.0)
    assert ddx.equation("-x ** 2").evaluate([3.0]) == pytest.approx(-9.0)
    # Every literal is a double, so this is a quotient and not a truncation.
    assert ddx.equation("(1/2) * x").evaluate([1.0]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("x ^ 2", "bad_syntax"),
        ("x > 0 ? x : -x", "bad_syntax"),
        ("x +", "bad_syntax"),
        ("", "bad_syntax"),
        ("nope(x)", "unknown_function"),
        ("sin(x, y)", "wrong_argument_count"),
        ("pow(x)", "wrong_argument_count"),
    ],
)
def test_refuses_with_the_code_that_says_why(source: str, code: str) -> None:
    with pytest.raises(ddx.Error) as caught:
        ddx.equation(source)
    assert caught.value.code is getattr(ddx.errc, code)
