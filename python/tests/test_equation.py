from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import Derivatives

import ddx


def test_values_gradient_and_hessian(
    scalar: ddx.Equation, expected: Derivatives
) -> None:
    assert scalar.evaluate([2.0, 3.0]) == pytest.approx(expected.f)

    value, gradient = scalar.jacobian([2.0, 3.0])
    assert value == pytest.approx(expected.f)
    assert gradient == pytest.approx([expected.dx, expected.dy])

    value, gradient, hessian = scalar.hessian([2.0, 3.0])
    assert hessian == pytest.approx(
        np.array([[expected.dxx, expected.dxy], [expected.dxy, expected.dyy]])
    )


def test_gradient_is_the_jacobian_without_the_value(
    scalar: ddx.Equation, expected: Derivatives
) -> None:
    """Its own lane, with no value block, and the Jacobian lane's answer to the bit."""
    gradient = scalar.gradient([2.0, 3.0])
    _, from_jacobian = scalar.jacobian([2.0, 3.0])
    assert gradient == pytest.approx([expected.dx, expected.dy])
    assert list(gradient) == list(from_jacobian)
    assert scalar.nodes(want=ddx.Want.GRADIENT) <= scalar.nodes()


def test_call_is_evaluate(scalar: ddx.Equation, expected: Derivatives) -> None:
    assert scalar([2.0, 3.0]) == pytest.approx(expected.f)


def test_the_three_point_spellings_agree(scalar: ddx.Equation) -> None:
    """Every spelling of a point names the same point.

    A dict is the keyword form C++ spells named<"x">(v), and a column array is
    the batch form at one point.
    """
    positional = scalar.jacobian([2.0, 3.0])
    by_name = scalar.jacobian({"x": 2.0, "y": 3.0})
    as_columns = scalar.jacobian(np.array([[2.0], [3.0]]))

    assert by_name[0] == pytest.approx(positional[0])
    assert by_name[1] == pytest.approx(positional[1])
    # The column spelling carries a points axis, so its answer keeps one.
    assert as_columns[0].shape == (1,)
    assert as_columns[1].shape == (2, 1)
    assert as_columns[1][:, 0] == pytest.approx(positional[1])


def test_a_system_has_a_runtime_output_count(system: ddx.Equation) -> None:
    assert system.outputs == 2
    assert system.arity == 2

    values, jacobian = system.jacobian([1.0, 2.0])
    # x^2 + y^2 - 4 and xy - 1 at (1, 2)
    assert values == pytest.approx([1.0, 1.0])
    # [[2x, 2y], [y, x]]
    assert jacobian == pytest.approx(np.array([[2.0, 4.0], [2.0, 1.0]]))


def test_a_system_still_has_a_hessian_at_a_point(system: ddx.Equation) -> None:
    _, _, hessian = system.hessian([1.0, 2.0])
    assert hessian.shape == (2, 2, 2)
    assert hessian[0] == pytest.approx(np.array([[2.0, 0.0], [0.0, 2.0]]))
    assert hessian[1] == pytest.approx(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_a_batched_hessian_needs_one_function(system: ddx.Equation) -> None:
    """A batched Hessian needs one function.

    A frozen graph carries one colouring, so there is no lane that could hold m
    of them.
    """
    batch = np.array([[1.0, 1.5], [2.0, 2.5]])
    with pytest.raises(ddx.Error) as caught:
        system.hessian(batch)
    assert caught.value.code is ddx.errc.wrong_column_count


def test_one_function_members_refuse_a_system(system: ddx.Equation) -> None:
    """H·v and the colour count are about one function's Hessian."""
    with pytest.raises(ddx.Error) as caught:
        system.hvp([1.0, 0.0], [1.0, 2.0])
    assert caught.value.code is ddx.errc.not_single_output
    with pytest.raises(ddx.Error) as colors:
        _ = system.hessian_colors
    assert colors.value.code is ddx.errc.not_single_output


def test_shapes_drop_the_axes_that_are_not_there(scalar: ddx.Equation) -> None:
    batch = np.array([[2.0, 2.5, 3.0], [3.0, 3.5, 4.0]])

    assert isinstance(scalar.evaluate([2.0, 3.0]), float)
    assert scalar.jacobian([2.0, 3.0])[1].shape == (2,)
    assert scalar.hessian([2.0, 3.0])[2].shape == (2, 2)

    assert scalar.evaluate(batch).shape == (3,)
    assert scalar.jacobian(batch)[1].shape == (2, 3)
    assert scalar.hessian(batch)[2].shape == (2, 2, 3)


def test_a_batch_agrees_with_the_points_in_it(scalar: ddx.Equation) -> None:
    batch = np.array([[2.0, 2.5, 3.0], [3.0, 3.5, 4.0]])
    _, columns = scalar.jacobian(batch)
    for i in range(batch.shape[1]):
        _, at_point = scalar.jacobian(batch[:, i])
        assert columns[:, i] == pytest.approx(at_point)


def test_a_dict_batch_needs_no_single_block(scalar: ddx.Equation) -> None:
    """Each value is its own column, so the arrays need not be one array."""
    xs = np.linspace(0.5, 2.0, 4)
    ys = np.linspace(0.1, 1.0, 4)
    loose = scalar.jacobian({"x": xs, "y": ys})
    packed = scalar.jacobian(np.stack([xs, ys]))
    assert loose[1] == pytest.approx(packed[1])


def test_a_scalar_in_a_dict_batch_is_held_at_every_point(scalar: ddx.Equation) -> None:
    xs = np.linspace(0.5, 2.0, 4)
    held = scalar.evaluate({"x": xs, "y": 0.5})
    spread = scalar.evaluate({"x": xs, "y": np.full(4, 0.5)})
    assert held == pytest.approx(spread)


def test_a_stray_name_is_not_a_missing_one(scalar: ddx.Equation) -> None:
    with pytest.raises(ddx.Error) as stray:
        scalar.evaluate({"x": 2.0, "z": 3.0})
    assert stray.value.code is ddx.errc.unknown_symbol

    with pytest.raises(ddx.Error) as missing:
        scalar.evaluate({"x": 2.0})
    assert missing.value.code is ddx.errc.wrong_arity


def test_a_point_of_the_wrong_length_is_refused(scalar: ddx.Equation) -> None:
    with pytest.raises(ddx.Error) as caught:
        scalar.evaluate([2.0])
    assert caught.value.code is ddx.errc.wrong_arity


def test_the_model_name_and_docstring_survive(scalar: ddx.Equation) -> None:
    assert scalar.__name__ == "eq"
    assert scalar.__doc__ is not None
    assert "exp(x)" in scalar.__doc__


def test_symbols_are_in_the_order_a_point_is_read_in(scalar: ddx.Equation) -> None:
    assert scalar.symbols == ["x", "y"]


def test_a_sparse_coupling_scatters_to_the_right_cells() -> None:
    """The compressed Hessian is indexed by colour and row, not as colours x n.

    ``exp(a) sin(b) + c*c*a`` needs three colours over three symbols and still
    leaves cells unowned -- b and c never couple -- so the block is shorter than
    colours x n.  Read as a dense grid it hands back another cell's value and
    then runs off the end.
    """
    e = math.exp(0.3)

    @ddx.equation
    def eq() -> ddx.Expression:
        """exp(a) sin(b) + c*c*a."""
        a, b, c = ddx.var("a"), ddx.var("b"), ddx.var("c")
        return ddx.exp(a) * ddx.sin(b) + c * c * a

    _, _, hessian = eq.hessian([0.3, 0.7, 1.1])

    assert eq.hessian_colors == 3
    assert hessian == pytest.approx(
        np.array(
            [
                [e * math.sin(0.7), e * math.cos(0.7), 2 * 1.1],
                [e * math.cos(0.7), -e * math.sin(0.7), 0.0],
                [2 * 1.1, 0.0, 2 * 0.3],
            ]
        )
    )


def test_symbols_take_the_order_they_are_named_in() -> None:
    def model():
        b = ddx.var("b")
        a = ddx.var("a")
        return a * a * b

    eq = ddx.equation(model)
    assert eq.symbols == ["b", "a"]
    assert eq([1.0, 2.0]) == 4.0  # b = 1, a = 2
    assert eq({"a": 2.0, "b": 1.0}) == 4.0


def test_a_param_is_an_input_with_no_column() -> None:
    def model():
        x = ddx.var("x")
        return ddx.param("a") * x * x

    eq = ddx.equation(model)
    assert eq.symbols == ["x"]
    assert eq.params == ["a"]
    assert eq.inputs == ["x", "a"]
    assert eq.arity == 1
    assert eq([3.0, 2.0]) == 18.0
    _, gradient = eq.jacobian({"x": 3.0, "a": 5.0})
    assert gradient.tolist() == [30.0]
    _, _, hv = eq.hvp([1.0], [3.0, 5.0])  # a direction is over the symbols only
    assert hv.tolist() == [10.0]
    with pytest.raises(ddx.Error) as missing:
        eq({"x": 3.0})
    assert missing.value.code is ddx.errc.wrong_arity


def test_a_name_is_one_kind_of_input() -> None:
    def model():
        x = ddx.var("x")
        return x * ddx.param("x")

    with pytest.raises(ddx.Error) as caught:
        ddx.equation(model)
    assert caught.value.code is ddx.errc.name_clash


def test_jacobian_jvp_moves_the_whole_jacobian_along_v() -> None:
    def model():
        x = ddx.var("x")
        y = ddx.var("y")
        return x * x * y, ddx.sin(x * y)

    eq = ddx.equation(model)
    px, py = 0.7, 1.3
    _, jacobian, moved = eq.jacobian_jvp([1.0, 0.0], [px, py])
    s, c = math.sin(px * py), math.cos(px * py)
    assert jacobian.shape == (2, 2)
    assert np.allclose(moved, [[2 * py, 2 * px], [-py * py * s, c - px * py * s]])


def test_restrict_keeps_the_chosen_columns() -> None:
    def model():
        t = ddx.var("T")
        x = ddx.var("x")
        p = ddx.var("P")
        return ddx.exp(t * x) + ddx.log(p) * x

    eq = ddx.equation(model)
    view = eq.restrict(["P", "T"])
    assert view.symbols == ["T", "P"]  # symbol order
    at = [1.5, 0.3, 2.0]
    _, full = eq.jacobian(at)
    _, part = view.jacobian(at)
    assert part.tolist() == [full[0], full[2]]
    _, _, moved = view.jacobian_jvp([1.0, 0.0, 0.0], at)
    _, _, full_moved = eq.jacobian_jvp([1.0, 0.0, 0.0], at)
    assert moved.tolist() == [full_moved[0], full_moved[2]]
    with pytest.raises(ddx.Error) as caught:
        eq.restrict(["nope"])
    assert caught.value.code is ddx.errc.unknown_symbol


def test_an_implicit_input_carries_dz_dx() -> None:
    def model():
        a = ddx.var("a")
        z = ddx.implicit("z")
        ddx.closes(z, z * z - a)
        return z

    eq = ddx.equation(model)
    assert eq.symbols == ["a"]
    assert eq.inputs == ["a", "z"]
    _, gradient = eq.jacobian([4.0, 2.0])
    assert gradient.tolist() == [0.25]
    _, _, hessian = eq.hessian({"a": 4.0, "z": 2.0})
    assert hessian.tolist() == [[-0.03125]]
    assert eq.residual([4.0, 3.0]) == [5.0]


def test_an_unclosed_implicit_is_refused() -> None:
    with pytest.raises(ddx.Error) as caught:
        ddx.equation(lambda: ddx.var("a") * ddx.implicit("z"))
    assert caught.value.code is ddx.errc.unclosed_implicit
