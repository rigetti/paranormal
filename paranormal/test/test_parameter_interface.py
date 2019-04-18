from enum import Enum
import json
import pytest

from paranormal.parameter_interface import *
from paranormal.params import *


class E(Enum):
    X = 0
    Y = 1


class P(Params):
    b = BoolParam(default=True, help='a boolean!')
    i = IntParam(default=1, help='an integer!')
    f = FloatParam(default=0.5, help='A float!')
    r = IntParam(required=True, help='An integer that is required!')
    e = EnumParam(cls=E, help='an enum', default=E.X)
    l = ListParam(subtype=int, default=[0, 1, 2], help='a list')
    a = ArangeParam(help='arange param', default=[0, 100, 5])


def test_params():

    class P(Params):
        b = BoolParam(default=True, help='a boolean!')
        i = IntParam(default=1, help='an integer!')
        f = FloatParam(default=0.5, help='A float!')
        r = IntParam(required=True, help='An integer that is required!')

    p = P(b=False, i=2, f=0.2, r=5)
    assert [(k, v) for k, v in p.items()] == [('b', False), ('i', 2), ('f', 0.2), ('r', 5)]

    # test with an argument that isn't in P
    with pytest.raises(KeyError):
        P(x=5, r=2)

    # Try without providing r
    with pytest.raises(KeyError):
        P()


def test_json_serialization():

    # to_json
    p = P(r=5)
    s = json.dumps(to_json_serializable_dict(p, include_defaults=False))
    assert s == '{"r": 5, "_type": "P", "_module": "test_parameter_interface"}'

    s = json.dumps(to_json_serializable_dict(p, include_defaults=True))
    assert s == '{"b": true, "i": 1, "f": 0.5, "r": 5, "e": "X", "l": [0, 1, 2], ' \
                '"a": [0, 100, 5], "_type": "P", "_module": "test_parameter_interface"}'
    # from json
    p = P(r=3)
    j = json.loads(json.dumps(to_json_serializable_dict(p, include_defaults=True)))
    assert p == from_json_serializable_dict(j)

    j = json.loads(json.dumps(to_json_serializable_dict(p, include_defaults=False)))
    assert p == from_json_serializable_dict(j)


def test_to_argparse():
    pass


def test_from_parsed_args():
    pass


def test_create_parser_and_parse_args():
    pass
