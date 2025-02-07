import hypothesis.strategies as st
import pytest
from hypothesis import given
from sympy import sqrt_mod

from cairo_addons.testing.strategies import felt
from cairo_addons.utils.uint384 import int_to_uint384, uint384_to_int
from cairo_ec.curve import AltBn128, Secp256k1

pytestmark = pytest.mark.python_vm


uint384 = st.integers(min_value=0, max_value=2**384 - 1)
curve = st.one_of(st.just(Secp256k1), st.just(AltBn128))


class TestEcOps:

    class TestTryGetPointFromX:
        @given(x=uint384, v=felt, curve=curve)
        def test_try_get_point_from_x(self, cairo_run, x, v, curve):
            y_try, is_on_curve = cairo_run(
                "test__try_get_point_from_x",
                x=int_to_uint384(x % curve.FIELD.PRIME),
                v=v,
                a=int_to_uint384(int(curve.A)),
                b=int_to_uint384(int(curve.B)),
                g=int_to_uint384(int(curve.G)),
                p=int_to_uint384(int(curve.FIELD.PRIME)),
            )

            square_root = sqrt_mod(x**3 + curve.A * x + curve.B, curve.FIELD.PRIME)
            assert (square_root is not None) == is_on_curve
            if square_root is not None:
                assert (
                    square_root
                    if (v % 2 == square_root % 2)
                    else (-square_root % curve.FIELD.PRIME)
                ) == uint384_to_int(y_try["d0"], y_try["d1"], y_try["d2"], y_try["d3"])

    class TestGetRandomPoint:
        @given(seed=felt, curve=curve)
        def test_should_return_a_point_on_the_curve(self, cairo_run, seed, curve):
            point = cairo_run(
                "test__get_random_point",
                seed=seed,
                a=int_to_uint384(int(curve.A)),
                b=int_to_uint384(int(curve.B)),
                g=int_to_uint384(int(curve.G)),
                p=int_to_uint384(int(curve.FIELD.PRIME)),
            )
            x = uint384_to_int(
                point["x"]["d0"],
                point["x"]["d1"],
                point["x"]["d2"],
                point["x"]["d3"],
            )
            y = uint384_to_int(
                point["y"]["d0"],
                point["y"]["d1"],
                point["y"]["d2"],
                point["y"]["d3"],
            )
            assert (
                x**3 + curve.A * x + curve.B
            ) % curve.FIELD.PRIME == y**2 % curve.FIELD.PRIME

    class TestEcAdd:
        @given(curve=curve)
        def test_ec_add(self, cairo_run, curve):
            p = curve.random_point()
            q = curve.random_point()
            res = cairo_run(
                "test__ec_add",
                p=[*int_to_uint384(int(p.x)), *int_to_uint384(int(p.y))],
                q=[*int_to_uint384(int(q.x)), *int_to_uint384(int(q.y))],
                a=int_to_uint384(int(curve.A)),
                g=int_to_uint384(int(curve.G)),
                modulus=int_to_uint384(int(curve.FIELD.PRIME)),
            )
            assert p + q == curve(
                *[curve.FIELD(uint384_to_int(**i)) for i in res.values()]
            )

        @given(curve=curve)
        def test_ec_add_equal(self, cairo_run, curve):
            p = curve.random_point()
            q = curve(p.x, p.y)
            res = cairo_run(
                "test__ec_add",
                p=[*int_to_uint384(int(p.x)), *int_to_uint384(int(p.y))],
                q=[*int_to_uint384(int(q.x)), *int_to_uint384(int(q.y))],
                a=int_to_uint384(int(curve.A)),
                g=int_to_uint384(int(curve.G)),
                modulus=int_to_uint384(int(curve.FIELD.PRIME)),
            )
            assert p + q == curve(
                *[curve.FIELD(uint384_to_int(**i)) for i in res.values()]
            )

        @given(curve=curve)
        def test_ec_add_opposite(self, cairo_run, curve):
            p = curve.random_point()
            q = curve(p.x, -p.y)
            res = cairo_run(
                "test__ec_add",
                p=[*int_to_uint384(int(p.x)), *int_to_uint384(int(p.y))],
                q=[*int_to_uint384(int(q.x)), *int_to_uint384(int(q.y))],
                a=int_to_uint384(int(curve.A)),
                g=int_to_uint384(int(curve.G)),
                modulus=int_to_uint384(int(curve.FIELD.PRIME)),
            )
            assert p + q == curve(
                *[curve.FIELD(uint384_to_int(**i)) for i in res.values()]
            )
