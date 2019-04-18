from enum import Enum

import numpy as np
import pytest

from paranormal.parameter_interface import BoolParam, Params, FloatParam, IntParam, StringParam, ListParam, SetParam, \
    EnumParam, GeomspaceParam, ArangeParam, SpanArangeParam


def test_bool_param():
    class MyParams(Params):
        param = BoolParam(help="A bool param", default=True)

    p = MyParams()

    assert p.param is True
    p.param = False
    assert p.param is False


def test_float_param():
    class MyParams(Params):
        param1 = FloatParam(help="A float param with no default and not required")

        param2 = FloatParam(help="A float param with units", default=4.0, unit="MHz")

    p = MyParams()

    assert p.param1 is None
    p.param1 = 3.0
    assert p.param1 == 3.0

    assert p.param2 == 4.0e6
    p.param2 = 5.0
    assert p.param2 == 5.0e6


def test_int_param():
    class MyParams(Params):
        param1 = IntParam(help="An int param that's required", required=True)

    with pytest.raises(KeyError):
        MyParams()

    p = MyParams(param1=4)
    assert p.param1 == 4


def test_string_param():
    class MyParams(Params):
        param1 = StringParam(help="A string param with a default", default="abc")

    p = MyParams()
    assert p.param1 == "abc"


def test_list_param():
    class MyParams(Params):
        param1 = ListParam(help="A list of ints", subtype=int, default=[1, 2, 3])

        param2 = ListParam(help="A list of floats with units", subtype=float, unit="MHz")

    p1 = MyParams()
    p2 = MyParams()

    assert p1.param1 == p2.param1 == [1, 2, 3]
    p1.param1 = [4, 5, 6]
    assert p1.param1 == [4, 5, 6]
    assert p2.param1 == [1, 2, 3]

    assert p1.param2 is None
    p1.param2 = [1.0, 2.0, 3.0]
    assert p1.param2 == [1.0e6, 2.0e6, 3.0e6]


def test_set_param():
    class MyParams(Params):
        param1 = SetParam(help="A set param")

    p = MyParams(param1={1, 2, 3, 3})

    assert p.param1 == {1, 2, 3}


def test_enum_param():
    class MyEnum(Enum):
        RED = 0
        GREEN = 1
        BLUE = 2

    class MyParams(Params):
        param1 = EnumParam(help="An enum param", cls=MyEnum, default=MyEnum.RED)

    p = MyParams()
    assert p.param1 == MyEnum.RED
    p.param1 = MyEnum.BLUE
    assert p.param1 == MyEnum.BLUE


def test_geomspace_param():
    class MyParams(Params):
        param1 = GeomspaceParam(help="A geometrically spaced list")

    p = MyParams(param1=[1, 2, 3])
    assert np.all(p.param1 == np.geomspace(1, 2, 3))


def test_arange_param():
    class MyParams(Params):
        param1 = ArangeParam(help="An evenly spaced list")

    p = MyParams(param1=[1, 2, 0.1])
    assert len(p.param1) == 10


def test_span_arange_param():
    class MyParams(Params):
        param1 = SpanArangeParam(help="An evenly spaced list created with center, width, step")

    p = MyParams(param1=[1, 2, 0.1])
    assert len(p.param1) == 20