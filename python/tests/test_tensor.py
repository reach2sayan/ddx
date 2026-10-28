"""Vectors in a model: one node per vector operation, the same numbers as the
scalar spelling."""

from __future__ import annotations

import math

import numpy as np
import pytest

import ddx

N = 5
M = [math.sin(1.0 + i * 7 + j * 3) for j in range(N) for i in range(N)]  # column-major


def m_at(i: int, j: int) -> float:
    return M[i + j * N]


def scalar_model() -> ddx.Expression:
    x = [ddx.var(f"x[{i}]") for i in range(N)]
    quad = sum(x[i] * m_at(i, j) * x[j] for i in range(N) for j in range(N))
    return quad + sum(xi * ddx.log(xi) for xi in x) + 2.0 * x[0] * x[1]


def tensor_model() -> ddx.Expression:
    x = ddx.vec("x", N)
    C = ddx.mat_constant(N, N, M)
    return ddx.dot(x, ddx.gemv(C, x)) + ddx.sum(x * ddx.log(x)) + 2.0 * x[0] * x[1]


POINT = [0.3 + 0.1 * i for i in range(N)]
DIRECTION = [math.cos(float(i)) for i in range(N)]


def test_the_vector_spelling_agrees_with_the_scalar_one() -> None:
    want = ddx.equation(scalar_model)
    got = ddx.equation(tensor_model)
    assert want.symbols == got.symbols
    assert math.isclose(want.evaluate(POINT), got.evaluate(POINT), rel_tol=1e-12)
    np.testing.assert_allclose(got.gradient(POINT), want.gradient(POINT), rtol=1e-12)
    _, _, want_hv = want.hvp(DIRECTION, POINT)
    _, _, got_hv = got.hvp(DIRECTION, POINT)
    np.testing.assert_allclose(got_hv, want_hv, rtol=1e-12)
    _, _, want_h = want.hessian(POINT)
    _, _, got_h = got.hessian(POINT)
    np.testing.assert_allclose(got_h, want_h, rtol=1e-12)


def test_matmul_is_dot_and_gemv_to_the_bit() -> None:
    def calls() -> ddx.Expression:
        x = ddx.vec("x", N)
        c = ddx.mat_constant(N, N, M)
        t = ddx.exp(0.1 * c)
        return (ddx.dot(x, ddx.gemv(c, x)) + ddx.dot(ddx.log(x), ddx.gemv(t, x, True))
                + ddx.dot(x, x))

    def operators() -> ddx.Expression:
        x = ddx.vec("x", N)
        c = ddx.mat_constant(N, N, M)
        t = ddx.exp(0.1 * c)
        return x @ (c @ x) + ddx.log(x) @ (ddx.transpose(t) @ x) + x @ x

    want = ddx.equation(calls)
    got = ddx.equation(operators)
    assert want.evaluate(POINT) == got.evaluate(POINT)
    np.testing.assert_array_equal(got.gradient(POINT), want.gradient(POINT))
    np.testing.assert_array_equal(got.hessian(POINT)[2], want.hessian(POINT)[2])


@pytest.mark.parametrize("product", [
    lambda x, c: c @ c,
    lambda x, c: x @ c,
    lambda x, c: ddx.vec("y", N + 1) @ x,
    lambda x, c: ddx.transpose(c) @ ddx.vec("y", N + 1),
])
def test_matmul_refuses_what_the_calls_refuse(product) -> None:  # noqa: ANN001
    with pytest.raises(ddx.Error) as refused:
        ddx.equation(lambda: product(ddx.vec("x", N), ddx.mat_constant(N, N, M)))
    assert refused.value.code == ddx.errc.tensor_shape


def test_select_over_a_vector_picks_the_arm_taken() -> None:
    def model() -> ddx.Expression:
        x = ddx.vec("x", 3)
        return ddx.sum(ddx.select(x > 0.0, x * ddx.log(x), 0.0))

    eq = ddx.equation(model)
    at = [0.0, 0.5, 2.0]
    grad = eq.gradient(at)
    assert grad[0] == 0.0
    np.testing.assert_allclose(grad[1:], [math.log(v) + 1.0 for v in at[1:]], rtol=1e-14)


def test_a_vector_knows_its_length_and_elements() -> None:
    def model() -> ddx.Expression:
        v = ddx.vec("v", 3)
        assert len(v) == 3
        assert v.shape == (3,)
        assert repr(v).startswith("Expression(pack")
        with pytest.raises(ddx.Error) as refused:
            _ = v[3]
        assert refused.value.code == ddx.errc.tensor_shape
        w = ddx.pack([v[2], v[1], v[0]])
        C = ddx.mat_constant(2, 3, [1.0] * 6)
        assert C.shape == (2, 3)
        assert v[0].shape == ()
        head = ddx.pack([v[0], v[1]])
        return ddx.dot(v, w) + ddx.sum(ddx.gemv(C, head, transposed=True))

    eq = ddx.equation(model)
    v = [1.0, 2.0, 3.0]
    assert math.isclose(eq.evaluate(v), 1 * 3 + 2 * 2 + 3 * 1 + 3 * (1 + 2))


def test_shapes_are_refused_where_they_are_formed() -> None:
    def mismatched() -> ddx.Expression:
        return ddx.dot(ddx.vec("a", 2), ddx.vec("b", 3))

    with pytest.raises(ddx.Error) as refused:
        ddx.equation(mismatched)
    assert refused.value.code == ddx.errc.tensor_shape

    with pytest.raises(ddx.Error) as outside:
        ddx.vec("x", 2)
    assert outside.value.code == ddx.errc.outside_equation


def test_a_scalar_scales_a_vector_and_literals_broadcast() -> None:
    def model() -> ddx.Expression:
        x = ddx.vec("x", 3)
        s = x[0] * x[1]
        return ddx.sum((2.0 * x + 1.0) * s - x / 2.0 + ddx.exp(-x) ** 2.0)

    eq = ddx.equation(model)
    x = [0.5, 1.5, -0.25]
    s = x[0] * x[1]
    want = sum((2.0 * xi + 1.0) * s - xi / 2.0 + math.exp(-xi) ** 2.0 for xi in x)
    assert math.isclose(eq.evaluate(x), want, rel_tol=1e-12)


def test_the_dense_hessian_is_the_coloured_one() -> None:
    eq = ddx.equation(tensor_model)
    _, g_want, h_want = eq.hessian(POINT)
    value, g, h = eq.dense_hessian(POINT)
    assert math.isclose(value, eq.evaluate(POINT), rel_tol=1e-12)
    np.testing.assert_allclose(g, g_want, rtol=1e-12)
    assert h.shape == (N, N)
    np.testing.assert_allclose(h, h_want, rtol=1e-10)


def test_a_matrix_of_temperature_matches_the_scalar_spelling() -> None:
    # sum_ij x_i x_j exp(T m_ij): exp(T * C) is a computed matrix, one node.
    def scalar() -> ddx.Expression:
        T = ddx.var("T")
        x = [ddx.var(f"x[{i}]") for i in range(N)]
        return sum(x[i] * ddx.exp(T * m_at(i, j)) * x[j]
                   for i in range(N) for j in range(N))

    def tensor() -> ddx.Expression:
        T = ddx.var("T")
        x = ddx.vec("x", N)
        tau = ddx.exp(T * ddx.mat_constant(N, N, M))
        assert tau.shape == (N, N)
        assert ddx.outer(x, x).shape == (N, N)
        return ddx.dot(x, ddx.gemv(tau, x))

    want = ddx.equation(scalar)
    got = ddx.equation(tensor)
    point = [1.3, *POINT]
    assert want.symbols == got.symbols
    np.testing.assert_allclose(got.gradient(point), want.gradient(point), rtol=1e-12)
    _, _, want_h = want.hessian(point)
    _, _, got_h = got.hessian(point)
    np.testing.assert_allclose(got_h, want_h, rtol=1e-12)


def test_released_tapes_grow_back_to_the_same_numbers() -> None:
    eq = ddx.equation(tensor_model)
    before = eq.gradient(POINT)
    ddx.release_tapes()
    np.testing.assert_array_equal(eq.gradient(POINT), before)


def test_a_param_vector_is_the_constant_it_carries() -> None:
    """vec_param's elements arrive with the point; nothing differentiates them."""
    c = [0.3, -0.2, 0.7]
    baked = ddx.equation(lambda: ddx.dot(ddx.vec_constant(c), ddx.log(ddx.vec("x", 3))))
    brought = ddx.equation(lambda: ddx.dot(ddx.vec_param("c", 3), ddx.log(ddx.vec("x", 3))))
    x = [0.2, 0.5, 0.3]
    assert list(brought.params) == ["c[0]", "c[1]", "c[2]"]
    assert np.array_equal(brought.jacobian(x + c)[1], baked.jacobian(x)[1])
    named = {f"x[{i}]": v for i, v in enumerate(x)} | {f"c[{i}]": v for i, v in enumerate(c)}
    assert brought.evaluate(named) == baked.evaluate(x)


def test_a_matrix_constant_takes_a_two_dimensional_array() -> None:
    """Any order, any float dtype: the shape is the array's, the data column-major."""
    m = np.arange(6.0).reshape(2, 3)
    x = [0.5, -1.0, 2.0]
    flat = ddx.equation(lambda: ddx.sum(ddx.mat_constant(2, 3, m.ravel(order="F")) @ ddx.vec("x", 3)))
    for given in (m, np.asfortranarray(m), m.astype(np.float32), m.tolist()):
        whole = ddx.equation(lambda given=given: ddx.sum(ddx.mat_constant(given) @ ddx.vec("x", 3)))
        assert whole.evaluate(x) == flat.evaluate(x)
    with pytest.raises(ddx.Error):
        ddx.equation(lambda: ddx.sum(ddx.mat_constant(np.ones(3)) @ ddx.vec("x", 3)))
