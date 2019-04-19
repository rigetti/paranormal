from argparse import Namespace
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
    class Colors(Enum):
        RED = 0
        BLUE = 1
        GREEN = 2
        YELLOW = 3

    class MySummer(Params):
        dpw_s = LinspaceParam(help='Drinks per weekend', expand=True, default=[0, None, 15],
                              unit='Drinks', prefix='s_')
        c = EnumParam(cls=Colors, default=Colors.BLUE, help='Preferred Drink color')
        t = FloatParam(help='Time it takes to finish a drink', default=60, unit='ns')
        f = FloatParam(help='Frequency of drinks on an average day', unit='MHz', default=None)
        do_something_crazy = BoolParam(default=False, help='Do something crazy')

    parser = to_argparse(MySummer)

    # parse with the defaults
    args = parser.parse_args([])
    # dpw was expanded and will still be in the namespace, just as None
    assert args == Namespace(c=Colors.BLUE, do_something_crazy=False, t=60, dpw_s=None,
                             s_start=0, s_stop=None, s_num=15, f=None)

    args = parser.parse_args(
        '--f 120 --c GREEN --s_start 20 --s_stop 600 --s_num 51 --do_something_crazy'.split(' '))
    assert args == Namespace(c='GREEN', do_something_crazy=True, t=60, dpw_s=None,
                             s_start=20, s_stop=600, s_num=51, f=120)

    # try with an argument that isn't part of the parser
    with pytest.raises(SystemExit):
        parser.parse_args(
            '--do_soooomething_crazy --f 120'.split(' '))

    class MyWinter(Params):
        s = FloatParam(default=12, help='hours sleeping per 24 hrs')
        hib = BoolParam(default=True, help='Whether or not to hibernate')
        dpw_w = LinspaceParam(help='Drinks per weekend', expand=True, default=[0, None, 15],
                              unit='Drinks', prefix='w_')

    class YearlySchedule(Params):
        winter = MyWinter()
        summer = MySummer(f=360)

    parser = to_argparse(YearlySchedule)







def test_from_parsed_args():
    pass


def test_create_parser_and_parse_args():
    pass
