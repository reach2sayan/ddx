from __future__ import annotations

import math
import operator
from collections.abc import Callable

import pytest

import ddx

# Every unary the library generates, against math's own.  The list is what the
# module exports, so an opcode added to the C++ tables is covered without this
# file changing.
UNARY = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": None,
    "atanh": math.atanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "erf": math.erf,
    "abs": abs,
    "neg": lambda v: -v,
    "sign": lambda v: math.copysign(1.0, v),
}


def test_every_exported_unary_is_covered() -> None:
    """Every unary the module exports is checked against math's own.

    The table above is written by hand and the C++ tables are not, so this is
    what fails when an opcode is added there and not here.
    """
    functions = {
        name
        for name in ddx.__all__
        if callable(getattr(ddx, name)) and not isinstance(getattr(ddx, name), type)
    }
    binary = {"add", "mul", "div", "pow", "atan2", "hypot", "max", "min",
              "lt", "le"}
    # `select` is the one ternary.
    ternary = {"select"}
    # Not operations at all: these build or read an equation.
    not_an_op = {"closes", "equation", "implicit", "param", "var"}
    # Vectors: factories and reductions, not scalar operations.
    vectors = {"add_at", "dot", "gemv", "gemv_t", "mat_const", "pack", "sum",
               "vec", "vec_const"}
    assert functions - binary - ternary - not_an_op - vectors == set(UNARY)


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary_matches_math(name: str) -> None:
    reference = UNARY[name]
    at = 1.5 if name == "acosh" else 0.4
    if reference is None:
        reference = math.acosh

    fn = getattr(ddx, name)
    eq = ddx.equation(lambda: fn(ddx.var("x")))
    assert eq.evaluate([at]) == pytest.approx(reference(at))


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("pow", lambda a, b: a**b),
        ("atan2", math.atan2),
        ("hypot", math.hypot),
        ("max", max),
        ("min", min),
        ("add", lambda a, b: a + b),
        ("mul", lambda a, b: a * b),
        ("div", lambda a, b: a / b),
    ],
)
def test_binary_matches_math(
    name: str, reference: Callable[[float, float], float]
) -> None:
    fn = getattr(ddx, name)
    eq = ddx.equation(lambda: fn(ddx.var("x"), ddx.var("y")))
    assert eq.evaluate([1.7, 2.3]) == pytest.approx(reference(1.7, 2.3))


def test_operators_and_reflection() -> None:
    eq = ddx.equation(lambda: (2.0 * ddx.var("x") + 1.0) / 4.0 - 0.5)
    assert eq.evaluate([3.0]) == pytest.approx((2.0 * 3.0 + 1.0) / 4.0 - 0.5)

    rhs = ddx.equation(lambda: 8.0 / ddx.var("x") - 2.0 ** ddx.var("x"))
    assert rhs.evaluate([3.0]) == pytest.approx(8.0 / 3.0 - 2.0**3.0)

    assert ddx.equation(lambda: abs(-ddx.var("x"))).evaluate([-2.0]) == 2.0


def test_interning_makes_a_repeated_subexpression_one_node() -> None:
    """A repeated subexpression is one node: an id *is* its identity."""

    @ddx.equation
    def shared() -> ddx.Expression:
        x, y = ddx.var("x"), ddx.var("y")
        return x * y + x * y

    assert shared.evaluate([2.0, 3.0]) == pytest.approx(12.0)
    assert shared.symbols == ["x", "y"]


def test_literals_fold_without_reaching_a_graph() -> None:
    """Two pending literals never become nodes, so this equation has no symbols."""
    eq = ddx.equation(lambda: ddx.Expression(2.0) * 3.0)
    assert eq.symbols == []
    assert eq.arity == 0
    assert eq.evaluate([]) == pytest.approx(6.0)


def test_var_outside_a_model_is_refused() -> None:
    with pytest.raises(ddx.Error) as caught:
        ddx.var("stray")
    assert caught.value.code is ddx.errc.no_arena


def test_a_raising_model_leaves_no_arena_installed() -> None:
    """A model that raises leaves no arena installed.

    The guard that restores the thread-local runs on the unwind; without it the
    next model would build into the arena this one abandoned.
    """

    def broken() -> ddx.Expression:
        ddx.var("x")
        raise ZeroDivisionError("from the model")

    with pytest.raises(ZeroDivisionError):
        ddx.equation(broken)

    with pytest.raises(ddx.Error) as caught:
        ddx.var("after")
    assert caught.value.code is ddx.errc.no_arena


def test_repr_names_what_it_is() -> None:
    eq = ddx.equation(lambda: ddx.var("x") * ddx.var("y"))
    assert repr(eq) == "<ddx.Equation (x, y) -> 1 output>"
    assert "Expression" in repr(ddx.Expression(2.0))


@pytest.mark.parametrize(
    ("compare", "a", "b", "want"),
    [
        (operator.lt, 1.0, 2.0, 1.0),
        (operator.lt, 2.0, 1.0, 0.0),
        (operator.lt, 2.0, 2.0, 0.0),
        (operator.le, 2.0, 2.0, 1.0),
        (operator.gt, 2.0, 1.0, 1.0),
        (operator.ge, 2.0, 2.0, 1.0),
        (operator.eq, 2.0, 2.0, 1.0),
        (operator.eq, 2.0, 1.0, 0.0),
        (operator.ne, 2.0, 1.0, 1.0),
    ],
)
def test_comparisons_answer_one_or_zero(
    compare: Callable[[ddx.Expression, ddx.Expression], ddx.Expression],
    a: float,
    b: float,
    want: float,
) -> None:
    """A comparison is a value, not a separate kind of thing."""
    eq = ddx.equation(lambda: compare(ddx.var("x"), ddx.var("y")))
    assert eq.evaluate([a, b]) == pytest.approx(want)


def test_a_number_compares_from_either_side() -> None:
    """`1.0 < x` reflects to `x > 1.0`; both are the same node."""
    left = ddx.equation(lambda: 1.0 < ddx.var("x"))  # noqa: SIM300  -- the reflected form is the point
    right = ddx.equation(lambda: ddx.var("x") > 1.0)
    for x in (0.5, 1.0, 2.0):
        assert left.evaluate([x]) == right.evaluate([x])
    assert isinstance(ddx.Expression(2.0) == 1.0, ddx.Expression)


def test_an_expression_has_no_truth_value() -> None:
    """A condition has no truth until a point is given.

    Python's own branch refuses rather than taking the first arm every time.
    """
    with pytest.raises(TypeError):
        bool(ddx.Expression(2.0) < 1.0)

    def branched() -> ddx.Expression:
        x = ddx.var("x")
        return x * x if x < 1.0 else 2.0 * x - 1.0

    with pytest.raises(TypeError):
        ddx.equation(branched)


def test_select_takes_the_arm_the_condition_names() -> None:
    """Both arms evaluate; the derivative is the taken arm's and nothing else."""
    eq = ddx.equation(
        lambda: ddx.select(
            ddx.var("x") < ddx.var("y"),
            ddx.var("x") * ddx.var("x"),
            ddx.var("y") ** 3,
        )
    )
    value, gradient = eq.jacobian([1.0, 3.0])
    assert value == pytest.approx(1.0)
    assert gradient == pytest.approx([2.0, 0.0])

    value, gradient = eq.jacobian([3.0, 1.0])
    assert value == pytest.approx(1.0)
    assert gradient == pytest.approx([0.0, 3.0])


def test_a_comparison_is_infix_in_text() -> None:
    """`<` is an operator in the grammar, where `select` is a call."""
    written = ddx.equation("select(x < y, x*x, y**3)")
    by_hand = ddx.equation(
        lambda: ddx.select(
            ddx.var("x") < ddx.var("y"),
            ddx.var("x") * ddx.var("x"),
            ddx.var("y") ** 3,
        )
    )
    assert written.jacobian([1.0, 3.0])[1] == pytest.approx(
        by_hand.jacobian([1.0, 3.0])[1]
    )
    assert ddx.equation("x != y").evaluate([1.0, 3.0]) == 1.0
    assert ddx.equation("x == y").evaluate([3.0, 3.0]) == 1.0
