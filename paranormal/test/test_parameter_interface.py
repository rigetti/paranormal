from argparse import Namespace
from enum import Enum
import json
import mock
import pytest
import tempfile

import numpy as np

from paranormal.parameter_interface import *
from paranormal.params import *


################################
# Helper Classes and Functions #
################################


class E(Enum):
    X = 0
    Y = 1


class P(Params):
    b = BoolParam(default=True, help='a boolean!')
    i = IntParam(default=1, help='an integer!')
    f = FloatParam(default=0.5e6, help='A float!', unit='MHz')
    r = IntParam(required=True, help='An integer that is required!')
    e = EnumParam(cls=E, help='an enum', default=E.X)
    l = ListParam(subtype=int, default=[0, 1, 2], help='a list')
    a = ArangeParam(help='arange param', default=(0, 100e-6, 5e-6), unit='us')


class Colors(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3


class MySummer(Params):
    dpw_s = LinspaceParam(help='Drinks per weekend', expand=True, default=(0, None, 15),
                          prefix='s_')
    c = EnumParam(cls=Colors, default=Colors.BLUE, help='Color of the sky')
    t = FloatParam(help='Time spent sunbathing before I burn', default=60e-9, unit='ns')
    f = FloatParam(help='Frequency of birds chirping', unit='MHz', default=None)
    do_something_crazy = BoolParam(default=False, help='Do something crazy')


class MyWinter(Params):
    s = FloatParam(default=12, help='hours sleeping per 24 hrs')
    hib = BoolParam(default=False, help='Whether or not to hibernate')
    dpw_w = LinspaceParam(help='Drinks per weekend', expand=True, default=(0, None, 15),
                          prefix='w_')


class MySpring(Params):
    flowers = FloatParam(default=12, help='flowers sprouting per day')
    dpw_s = LinspaceParam(help='Drinks per weekend', expand=True, default=(0, None, 15),
                          prefix='sp_')


class PositionalsA(Params):
    x = FloatParam(positional=True, help='a positional float')
    y = StringParam(positional=True, help='a positional string')
    z = LinspaceParam(positional=True, help='a positional linspace')


class PositionalsB(Params):
    a = IntParam(positional=True, help='a positional int')
    b = IntParam(help='an int')


class FreqSweep(Params):
    freqs = LinspaceParam(help='freqs', expand=True, default=(10e6, 20e6, 30), unit='MHz',
                          prefix='f_')
    times = LinspaceParam(help='times', expand=True, default=(100e-6, 200e-6, 50), unit='us',
                          prefix='t_')


class TimeSweep(Params):
    times = LinspaceParam(help='times', expand=True, default=(100e-9, 500e-9, 20), unit='ns',
                          prefix='t_')


class DoubleSweep(Params):
    freq_sweep = FreqSweep()
    time_sweep = TimeSweep()


class HiddenWinterSchedule(Params):
    winter = MyWinter(s='__hide__', dpw_w='__hide__')


class HiddenFall(Params):
    leaf = FloatParam(default=2, help='number of falling leaves caught', hide=True)
    pumpkin = BoolParam(default=True, help='Whether the pumpkin got carved', hide=False)


def _compare_two_param_item_lists(a, b):
    for (k, v), (k_cor, v_cor) in zip(a, b):
        assert k == k_cor
        if isinstance(v, np.ndarray):
            assert np.allclose(v, v_cor)
        elif isinstance(v, float) or isinstance(v, int):
            assert np.isclose(v, v_cor)
        elif isinstance(v, Params):
            _compare_two_param_item_lists(v.items(), v_cor.items())
        else:
            assert v == v_cor


#####################
# Actual Unit Tests #
#####################


def test_params():
    p = P(b=False, i=2, f=0.2e6, r=5)
    correct_values = [('b', False), ('i', 2), ('f', 0.2e6), ('r', 5), ('e', E.X), ('l', [0, 1, 2]),
                      ('a', np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
                                      75, 80, 85, 90, 95]) * 1e-6)]
    _compare_two_param_item_lists(p.items(), correct_values)

    # test with an argument that isn't in P
    with pytest.raises(KeyError):
        P(x=5, r=2)

    # Try without providing r
    with pytest.raises(KeyError):
        P()

    # test non_si_set
    p.unit_set('f', 0.3)
    assert p.f == 0.3e6
    p.unit_set('a', [1.2, 1.5, 100])
    assert np.allclose(p.a, np.arange(1.2, 1.5, 100) * 1e-6)


def test_json_serialization():
    # to_json
    p = P(r=5)
    s = json.dumps(to_json_serializable_dict(p, include_defaults=False))
    assert s == '{"r": 5, "_type": "P", "_module": "test_parameter_interface"}'

    s = json.dumps(to_json_serializable_dict(p, include_defaults=True))
    assert s == '{"b": true, "i": 1, "f": 500000.0, "r": 5, "e": "X", "l": [0, 1, 2], ' \
                '"a": [0, 0.0001, 5e-06], "_type": "P", "_module": "test_parameter_interface"}'
    # from json
    p = P(r=3)
    j = json.loads(json.dumps(to_json_serializable_dict(p, include_defaults=True)))
    assert p == from_json_serializable_dict(j)

    j = json.loads(json.dumps(to_json_serializable_dict(p, include_defaults=False)))
    assert p == from_json_serializable_dict(j)

    # test with hiding some nested params
    y = HiddenWinterSchedule()
    j = json.loads(json.dumps(to_json_serializable_dict(y)))
    assert y == from_json_serializable_dict(j)

    # test with setting nested params
    y = DoubleSweep()
    y.freq_sweep.freqs = (100, 200, 300)
    s = json.dumps(to_json_serializable_dict(y))
    assert s == '{"freq_sweep": {"freqs": [100, 200, 300], "times": [0.0001, 0.0002, 50], ' \
                '"_type": "FreqSweep", "_module": "test_parameter_interface"}, ' \
                '"time_sweep": {"times": [1e-07, 5e-07, 20], "_type": "TimeSweep", ' \
                '"_module": "test_parameter_interface"}, "_type": "DoubleSweep", ' \
                '"_module": "test_parameter_interface"}'

    # test changing defaults of nested params in one class instance doesn't change the default
    # for all class instances
    x = DoubleSweep()
    assert x != y

    # ensure warnings are working when setting something to __hide__
    y.freq_sweep.freqs = '__hide__'
    with pytest.warns(UserWarning):
        to_json_serializable_dict(y)


def test_yaml_serialization():
    p = P(r=10)
    temp = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
    with mock.patch('paranormal.parameter_interface.open') as o:
        o.return_value = temp
        to_yaml_file(p, 'mock.yaml', include_defaults=True)
    with open(temp.name) as yaml_file:
        assert yaml_file.read() == 'b: true\ni: 1\nf: 500000.0\nr: 10\ne: X\nl:\n- 0\n- 1\n- 2\n' \
                                   'a: !!python/tuple\n- 0\n- 0.0001\n- 5.0e-06\n_type: P\n' \
                                   '_module: test_parameter_interface\n'

    # test yaml dumping with alphabetical reorder
    p = P(r=10)
    temp = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
    with mock.patch('paranormal.parameter_interface.open') as o:
        o.return_value = temp
        to_yaml_file(p, 'mock.yaml', include_defaults=True, sort_keys=True)
    with open(temp.name) as yaml_file:
        assert yaml_file.read() == '_module: test_parameter_interface\n_type: P\na: ' \
                                   '!!python/tuple\n- 0\n- 0.0001' \
                                   '\n- 5.0e-06\nb: true\ne: X\nf: 500000.0\ni: 1\n' \
                                   'l:\n- 0\n- 1\n- 2\nr: 10\n'


def test_merge_params():
    merged_positionals = merge_param_classes(PositionalsA, PositionalsB)
    for attr in PositionalsA():
        if getattr(PositionalsA.__dict__.get(attr), 'positional', False):
            assert merged_positionals.__dict__.get(attr, None) is None
        else:
            assert merged_positionals.__dict__.get(attr, None) == PositionalsA.__dict__.get(attr)
    for attr in PositionalsB():
        if getattr(PositionalsB.__dict__.get(attr), 'positional', False):
            assert merged_positionals.__dict__.get(attr, None) is None
        else:
            assert merged_positionals.__dict__.get(attr, None) == PositionalsB.__dict__.get(attr)

    # try without merging the positionals
    merged_positionals = merge_param_classes(PositionalsA, PositionalsB,
                                             merge_positional_params=False)
    for attr in PositionalsA():
        assert merged_positionals.__dict__.get(attr, None) == PositionalsA.__dict__.get(attr)
    for attr in PositionalsB():
        assert merged_positionals.__dict__.get(attr, None) == PositionalsB.__dict__.get(attr)

    # try with conflicting parameters
    with pytest.raises(ValueError):
        merge_param_classes(FreqSweep, TimeSweep)

    # try with nested classes
    class YearlySchedule(Params):
        winter = MyWinter()
        summer = MySummer()
        x = IntParam(help='an int')

    merged = merge_param_classes(DoubleSweep, YearlySchedule,
                                 merge_positional_params=False)
    for attr in YearlySchedule():
        assert merged.__dict__.get(attr, None) == YearlySchedule.__dict__.get(attr)
    for attr in DoubleSweep():
        assert merged.__dict__.get(attr, None) == DoubleSweep.__dict__.get(attr)


def test_append_params_attributes():
    class A(Params):
        i = IntParam(help='int', default=1)

    class B(Params):
        f = FloatParam(help='float', default=2.0)
        z = IntParam(help='another int')

    append_params_attributes(A, B, do_not_copy=['z'], override_dictionary={
        'f': {'help': 'a better float', 'default': 3.0, 'positional': True}})
    a = A()
    assert a.f == 3.0
    assert A.__dict__['f'].help == 'a better float'
    assert A.__dict__['f'].positional

    with pytest.raises(AttributeError):
        assert a.z


def test_to_argparse():
    parser = to_argparse(MySummer)

    # parse with the defaults
    args = parser.parse_args([])
    # dpw was expanded and will still be in the namespace, just as None
    ns = Namespace(c=Colors.BLUE, do_something_crazy=False, t=60, dpw_s=None,
                   s_start=0, s_stop=None, s_num=15, f=None)
    assert isinstance(args, Namespace)
    _compare_two_param_item_lists(sorted(ns.__dict__.items()), sorted(args.__dict__.items()))

    args = parser.parse_args(
        '--f 120 --c GREEN --s_start 20 --s_stop 600 --s_num 51 --do_something_crazy'.split(' '))
    ns = Namespace(c='GREEN', do_something_crazy=True, t=60, dpw_s=None,
                   s_start=20, s_stop=600, s_num=51, f=120)
    assert isinstance(args, Namespace)
    _compare_two_param_item_lists(sorted(ns.__dict__.items()), sorted(args.__dict__.items()))

    # try with an argument that isn't part of the parser
    with pytest.raises(SystemExit):
        parser.parse_args(
            '--do_soooomething_crazy --f 120'.split(' '))

    class YearlySchedule(Params):
        winter = MyWinter()
        summer = MySummer(f=360e6)

    parser = to_argparse(YearlySchedule)
    args = parser.parse_args(
        '--summer_c RED --winter_s 22 --summer_s_start 20 --summer_s_stop 600 --winter_w_start 20 '
        '--winter_w_stop 200 --winter_hib'.split(' '))
    ns = Namespace(summer_c='RED', summer_do_something_crazy=False, summer_dpw_s=None,
                   summer_f=360, summer_s_num=15, summer_s_start=20.0,
                   summer_s_stop=600.0, summer_t=60,
                   winter_dpw_w=None, winter_hib=True, winter_s=22.0, winter_w_num=15,
                   winter_w_start=20.0, winter_w_stop=200.0)
    assert isinstance(args, Namespace)
    _compare_two_param_item_lists(sorted(ns.__dict__.items()), sorted(args.__dict__.items()))

    class YearlySchedule(Params):
        winter = MyWinter()
        summer = MySummer()
        spring = MySpring()

    parser = to_argparse(YearlySchedule)
    args = parser.parse_args([])
    ns = Namespace(summer_c=Colors.BLUE, summer_do_something_crazy=False,
                   summer_dpw_s=None, summer_f=None, summer_s_num=15, summer_s_start=0,
                   summer_s_stop=None, summer_t=60., winter_dpw_w=None, winter_hib=False,
                   winter_s=12, winter_w_num=15, winter_w_start=0, winter_w_stop=None,
                   spring_dpw_s=None, spring_flowers=12, spring_sp_num=15,
                   spring_sp_start=0, spring_sp_stop=None)
    assert isinstance(args, Namespace)
    _compare_two_param_item_lists(sorted(ns.__dict__.items()), sorted(args.__dict__.items()))

    # Make sure conflicting params are resolved
    parser = to_argparse(DoubleSweep)
    args = parser.parse_args([])
    ns = Namespace(freq_sweep_f_num=30, freq_sweep_f_start=10, freq_sweep_f_stop=20,
                   freq_sweep_freqs=None, freq_sweep_t_num=50, freq_sweep_t_start=100,
                   freq_sweep_t_stop=200, freq_sweep_times=None, time_sweep_t_num=20,
                   time_sweep_t_start=100, time_sweep_t_stop=500, time_sweep_times=None)
    assert isinstance(args, Namespace)
    _compare_two_param_item_lists(sorted(ns.__dict__.items()), sorted(args.__dict__.items()))

    # make sure check that requires prefixes if expand=True for multiple classes is working
    class BadFreqSweep(Params):
        freqs = LinspaceParam(help='freqs', expand=True)
        times = LinspaceParam(help='times', expand=True)

    with pytest.raises(ValueError):
        to_argparse(BadFreqSweep)

    # test with hiding some nested params
    parser = to_argparse(HiddenWinterSchedule)
    args = parser.parse_args([])
    assert vars(args) == {'winter_hib': False}

    # test by overriding prefix behavior
    class PrefixMania(Params):
        x = ArangeParam(help='some arange', default=(0, 10, 20))

    class NestedPrefixMania(Params):
        a = PrefixMania()
        b = PrefixMania()
        __nested_prefixes__ = {'a': 'ayy', 'b': None}

    parser = to_argparse(NestedPrefixMania)
    args = parser.parse_args([])
    assert vars(args) == {'ayy_x': (0, 10, 20), 'x': (0, 10, 20)}


def test_from_parsed_args():
    parser = to_argparse(MySummer)
    y = from_parsed_args(MySummer, params_namespace=parser.parse_args([]))[0]
    correct_items = [('dpw_s', [0, None, 15]), ('c', Colors.BLUE), ('t', 60e-9), ('f', None),
                     ('do_something_crazy', False)]
    _compare_two_param_item_lists(y.items(), correct_items)

    args = parser.parse_args(
        '--s_start .1 --s_stop 2 --s_num 10 --t 20 --do_something_crazy'.split(' '))
    y = from_parsed_args(MySummer, params_namespace=args)[0]
    correct_items = [('dpw_s', list(np.linspace(.1, 2, 10))), ('c', Colors.BLUE), ('t', 20e-9),
                     ('f', None), ('do_something_crazy', True)]
    _compare_two_param_item_lists(y.items(), correct_items)

    class YearlySchedule(Params):
        winter = MyWinter()
        summer = MySummer(f=None)

    # test that nested classes work
    parser = to_argparse(YearlySchedule)
    args = parser.parse_args(
        '--summer_c RED --winter_s 22 --summer_s_start 20 --summer_s_stop 600 --winter_w_start 20 '
        '--winter_w_stop 200 --winter_hib'.split(' '))
    y = from_parsed_args(YearlySchedule, params_namespace=args)[0]
    correct_items = [('winter', MyWinter(s=22, dpw_w=[20, 200, 15], hib=True)),
                     ('summer', MySummer(c=Colors.RED, dpw_s=[20, 600, 15]))]
    _compare_two_param_item_lists(y.items(), correct_items)

    # test that nested classes with positionals work
    class PositionalsC(Params):
        a_pos = PositionalsA()
        b_pos = PositionalsB()

    parser = to_argparse(PositionalsC)
    args = parser.parse_args(
        '1.0 hey 0.0 1.0 22.0 1 --b 10'.split(' '))
    y = from_parsed_args(PositionalsC, params_namespace=args)[0]
    correct_items = [('a_pos', PositionalsA(x=1.0, z=[0.0, 1.0, 22.0], y='hey')),
                     ('b_pos', PositionalsB(a=1, b=10))]
    _compare_two_param_item_lists(y.items(), correct_items)

    parser = to_argparse(DoubleSweep)

    # make sure if you pass an expanded param, an error is thrown
    with pytest.raises(AssertionError):
        args = parser.parse_args([])
        setattr(args, 'freq_sweep_times', [0, 100, 200])
        from_parsed_args(DoubleSweep, params_namespace=args)

    args = parser.parse_args(
        '--time_sweep_t_start 20 --time_sweep_t_stop 30 --freq_sweep_f_stop 40'.split(' '))
    y = from_parsed_args(DoubleSweep, params_namespace=args)[0]
    correct_items = [('freq_sweep', FreqSweep(freqs=[10e6, 40.0e6, 30])),
                     ('time_sweep', TimeSweep(times=[20e-9, 30e-9, 20]))]
    _compare_two_param_item_lists(y.items(), correct_items)

    # test with hiding some nested params
    parser = to_argparse(HiddenWinterSchedule)
    args = parser.parse_args([])
    y = from_parsed_args(HiddenWinterSchedule, params_namespace=args)[0]
    correct_items = [('winter', MyWinter(dpw_w=None, s=None))]
    _compare_two_param_item_lists(y.items(), correct_items)

    # test that hide param works
    parser = to_argparse(HiddenFall)
    help = parser.format_help()
    assert 'pumpkin' in help
    assert 'leaf' not in help


def test_create_parser_and_parse_args():
    # test that nested classes with positionals work
    class PositionalsC(Params):
        a_pos = PositionalsA()
        b_pos = PositionalsB()

    correct_items = [('a_pos', PositionalsA(x=1.0, z=[0.0, 1.0, 22.0], y='hey')),
                     ('b_pos', PositionalsB(a=1, b=10))]
    with mock.patch('paranormal.parameter_interface.ArgumentParser.parse_known_args') as pa:
        pa.return_value = (Namespace(b=10, positionals=['1.0', 'hey', '0.0', '1.0', '22.0', '1']),
                           [])
        y = create_parser_and_parse_args(PositionalsC)
    _compare_two_param_item_lists(y.items(), correct_items)

    # test that nested classes work:
    class YearlySchedule(Params):
        winter = MyWinter()
        summer = MySummer(f=None)
        __nested_prefixes__ = {'winter': None, 'summer': None}

    with mock.patch('paranormal.parameter_interface.ArgumentParser.parse_known_args') as pa:
        pa.return_value = (Namespace(c='RED', do_something_crazy=False, dpw_s=None, dpw_w=None,
                                     f=None, hib=True, s=22.0, s_num=15, s_start=20.0, s_stop=600.0,
                                     t=60, w_num=15, w_start=20.0, w_stop=200.0), [])
        y = create_parser_and_parse_args(YearlySchedule)
    correct_items = [('winter', MyWinter(s=22, dpw_w=[20, 200, 15], hib=True)),
                     ('summer', MySummer(c=Colors.RED, dpw_s=[20, 600, 15]))]
    _compare_two_param_item_lists(y.items(), correct_items)
