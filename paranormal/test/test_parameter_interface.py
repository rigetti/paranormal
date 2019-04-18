from enum import Enum
import json
import pytest
import re

from pampy import MatchError

from paranormal.parameter_interface import *
from paranormal.parameter_interface import (
    _add_param_to_parser,
    _check_for_required_arguments,
    _convert_type_to_regex,
    _expand_multi_arg_param,
    _extract_expanded_param,
    _flatten_cls_params,
    _merge_param_classes,
    _parse_positional_arguments)
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


def test_check_for_required_arguments():
    class P(Params):
        i = IntParam(required=True, help='a mandatory int!')
    _check_for_required_arguments(P, {'i': 5})

    with pytest.raises(KeyError):
        _check_for_required_arguments(P, {'x': 3})


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


@pytest.mark.skip(msg='Inheritance is broken')
def test_merge_param_classes():

    class MILLER(Params):
        x = FloatParam(positional=True, help='A positional float')
        y = IntParam(positional=True, help='a positional int')

    class PBR(P):
        z = ListParam(positional=True, help='A positional list of ints', subtype=int)

    COORS = _merge_param_classes([MILLER, PBR])
    c = COORS(positionals=[1, 2, 3, 0.5], r=5)
    for k in ['b', 'i', 'f', 'r', 'e', 'l', 'a', 'positionals']:
        assert getattr(c, k, None) is not None

    # test without merging positionals
    COORS = _merge_param_classes([MILLER, PBR], merge_positional_params=False)
    with pytest.raises(KeyError):
        COORS(r=5, positionals=[1,2,3,4])

    # test with parameter conflict
    class IPA(P):
        pass
    with pytest.raises(ValueError):
        _merge_param_classes([IPA, PBR])

    # test with only a single positional (merging positionals shouldn't happen, regardless of flag)
    class BUD(Params):
        q = BoolParam(help='Another bool', default=True)
    COORS = _merge_param_classes([PBR, BUD])
    with pytest.raises(KeyError):
        COORS(r=5, positionals=[1,2,3,4])


def test_convert_type_to_regex():
    int_re = _convert_type_to_regex(int)
    float_re = _convert_type_to_regex(float)
    string_re = _convert_type_to_regex(str)
    with pytest.raises(KeyError):
        _convert_type_to_regex(dict)
    assert re.match(int_re, '1') is not None
    assert re.match(int_re, '1.2') is None
    assert re.match(int_re, 'hey') is None

    assert re.match(float_re, '1') is None
    assert re.match(float_re, '1.2') is not None
    assert re.match(float_re, '.2') is not None
    assert re.match(float_re, '1.') is not None
    assert re.match(float_re, '1.2e-6') is not None
    assert re.match(float_re, '2.34E6') is not None
    assert re.match(float_re, '1..2') is None
    assert re.match(float_re, 'hey') is None

    assert re.match(string_re, '1') is not None
    assert re.match(string_re, '1.2') is not None
    assert re.match(string_re, 'hey') is not None
    assert re.findall(string_re, 'hey what is up') == ['hey', 'what', 'is', 'up']


@pytest.mark.skip(msg='Positional parser is broken')
def test_parse_positional_arguments():
    positionals = {'i': IntParam(positional=True, choices=[1, 2, 3], help='an int'),
                   'f': FloatParam(positional=True, help='a float'),
                   's': StringParam(positional=True, help='a string'),
                   'l': ListParam(positional=True, subtype=str, help='a list of strings')}
    list_of_positionals = ['1', '4.0', '5', '4.5', 'hey', 'a string']
    import pdb; pdb.set_trace()
    matched = _parse_positional_arguments(list_of_positionals, positionals)


def test_add_param_to_parser():
    pass


def test_expand_multi_arg_param():
    pass


def test_extract_expanded_param():
    pass


def test_flatten_cls_params():
    pass


def test_to_argparse():
    pass


def test_from_parsed_args():
    pass


def test_create_parser_and_parse_args():
    pass
