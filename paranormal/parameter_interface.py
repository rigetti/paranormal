from abc import ABC
from argparse import ArgumentParser, Namespace, SUPPRESS
import copy
from collections import Mapping
from enum import Enum, EnumMeta
import importlib
import re
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import yaml

import numpy as np
from pampy import match, _


__all__ = ['BoolParam', 'FloatParam', 'IntParam', 'StringParam', 'ListParam', 'SetParam',
           'ArangeParam', 'EnumParam', 'GeomspaceParam', 'Params', 'to_json_serializable_dict',
           'from_json_serializable_dict', 'to_yaml_file', 'from_yaml_file',
           'create_parser_and_parser_args', 'to_argparse', 'convert_to_si_units', 'BaseParams',
           'meas_config_to_base_params']


#################################
# All the Different Param Types #
#################################

class BaseDescriptor(ABC):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name, None)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name

    def to_json(self, instance, include_default: bool = True):
        return instance.__dict__.get(self.name,
                                     getattr(self, 'default', None) if include_default else None)


class BoolParam(BaseDescriptor):
    def __init__(self, *, help: str, default: bool, **kwargs):
        if 'required' in kwargs or default is None:
            raise ValueError('BoolParam cannot be a required argument '
                             'and must have a default value')
        self.default = default
        self.help = help
        super(BoolParam, self).__init__(**kwargs)


    def __get__(self, instance, owner) -> bool:
        return instance.__dict__.get(self.name, self.default)


class FloatParam(BaseDescriptor):
    def __init__(self, *,
                 help: str,
                 default: Optional[float] = None,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.required = required
        self.help = help
        self.unit = unit
        super(FloatParam, self).__init__(**kwargs)


    def __get__(self, instance, owner) -> Optional[float]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        if instance.__dict__.get(self.name, self.default) is None:
            return None
        return convert_to_si_units(instance.__dict__.get(self.name, self.default), self.unit)


class IntParam(BaseDescriptor):
    def __init__(self, *,
                 help: str,
                 default: Optional[int] = None,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.required = required
        self.help = help
        self.unit = unit
        super(IntParam, self).__init__(**kwargs)


    def __get__(self, instance, owner) -> Optional[int]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        if instance.__dict__.get(self.name, self.default) is None:
            return None
        return convert_to_si_units(instance.__dict__.get(self.name, self.default), self.unit)


class StringParam(BaseDescriptor):
    def __init__(self, *,
                 help: str,
                 default: Optional[str] = None,
                 required: Optional[bool] = None,
                 **kwargs):
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.required = required
        self.help = help
        super(StringParam, self).__init__(**kwargs)


    def __get__(self, instance, owner) -> Optional[str]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        if instance.__dict__.get(self.name, self.default) is None:
            return None
        return instance.__dict__.get(self.name, self.default)


class ListParam(BaseDescriptor):
    def __init__(self, *,
                 help: str,
                 subtype: type = str,
                 default: Optional[List] = None,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.required = required
        self.help = help
        self.unit = unit
        self.subtype = subtype
        super(ListParam, self).__init__(**kwargs)


    def __get__(self, instance, owner) -> Optional[List]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        if instance.__dict__.get(self.name, self.default) is None:
            return None
        return list(convert_to_si_units(v, self.unit)
                    for v in instance.__dict__.get(self.name, self.default))


class SetParam(BaseDescriptor):
    def __init__(self, *,
                 help: str,
                 subtype: type = str,
                 default: Optional[Set] = None,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.required = required
        self.help = help
        self.unit = unit
        self.subtype = subtype
        super(SetParam, self).__init__(**kwargs)


    def __get__(self, instance, owner) -> Optional[Set]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        if instance.__dict__.get(self.name, self.default) is None:
            return None
        return set(convert_to_si_units(v, self.unit)
                   for v in instance.__dict__.get(self.name, self.default))


class EnumParam(BaseDescriptor):
    def __init__(self, *,
                 help: str,
                 cls: EnumMeta,
                 default: Optional[Enum] = None,
                 required: Optional[bool] = None,
                 **kwargs):
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.cls = cls
        self.default = default
        self.required = required
        self.help = help
        super(EnumParam, self).__init__(**kwargs)


    def __get__(self, instance, owner) -> Optional[Enum]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        if instance.__dict__.get(self.name, self.default) is None:
            return None
        return self.cls[instance.__dict__.get(self.name, self.default)]

    def __set__(self, instance, value):
        if value in self.cls:
            instance.__dict__[self.name] = value.name
        elif value in self.cls.__members__:
            instance.__dict__[self.name] = value
        else:
            raise KeyError(f'{value} is not a valid member of {self.cls.__name__}')


class GeomspaceParam(BaseDescriptor):
    nargs = 3
    def __init__(self, *,
                 help: str,
                 default: Optional[List] = None,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.required = required
        self.help = help
        self.unit = unit
        super(GeomspaceParam, self).__init__(**kwargs)


    def __get__(self, instance, owner) -> Optional[np.ndarray]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        if instance.__dict__.get(self.name, self.default) is None:
            return None
        return convert_to_si_units(np.geomspace(*instance.__dict__.get(self.name, self.default)),
                                   self.unit)

    def __set__(self, instance, value):
        assert (isinstance(value, list) and value[0] != 0 and len(value) == 3) or value is None
        instance.__dict__[self.name] = value


class ArangeParam(BaseDescriptor):
    nargs = 3
    def __init__(self, *,
                 help: str,
                 default: Optional[List] = None,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.required = required
        self.help = help
        self.unit = unit
        super(ArangeParam, self).__init__(**kwargs)

    def __get__(self, instance, owner) -> Optional[np.ndarray]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        if instance.__dict__.get(self.name, self.default) is None:
            return None
        return convert_to_si_units(np.arange(*instance.__dict__.get(self.name, self.default)),
                                   self.unit)

    def __set__(self, instance, value):
        assert (isinstance(value, list) and len(value) == 3) or value is None
        instance.__dict__[self.name] = value


###################
# Unit Conversion #
###################


UNIT_CONVERSION_TABLE = {"GHz" : 1.0e9,
                         "MHz": 1.0e6,
                         "ns": 1.0e-9,
                         "μs": 1.0e-6,
                         "mV": 1.0e-3,
                         "mW": 1.0e-3,
                         "dBm": 1,
                         "V": 1,
                         "Hz": 1,
                         "s": 1,
                         "Scalar": 1}


def convert_to_si_units(value, unit: Optional[str] = None):
    """
    Convert value to SI units from specified unit.
    """
    if unit is None:
        return value
    try:
        return value * UNIT_CONVERSION_TABLE[unit]
    except KeyError:
        raise KeyError(f'Unit "{unit}" not in conversion table!')
    except TypeError:
        raise TypeError(f'Unit "{unit}" not compatible with value {value} of type {type(value)}')


####################
# Params Interface #
####################


class Params(Mapping):
    """
    Base class for Params - supports:

    1. ** operation
    2 * operation (iterate through param names)

    Inherit from this class and populate with Param classes from above. Ex.
    ```
    class A(Params):
        g = GeomspaceParam(help='test', default=[0, 100, 20])
        i = IntParam(help='my favorite int', required=True, choices=[1,2,3])
    ```
    """
    def __init__(self, **kwargs):
        cls = type(self)
        for key, value in kwargs.items():
            if not key in cls.__dict__:
                raise TypeError(f'{cls.__name__} does not take {key} as an argument')
            setattr(self, key, value)
        _check_for_required_arguments(cls, kwargs)

    def __iter__(self):
        valid_keys = [k for k in self.__class__.__dict__.keys() if not k.startswith('_')]
        for key in valid_keys:
            yield key

    def __len__(self):
        return len(list(self.__iter__()))

    def __getitem__(self, item):
        return getattr(self, item)


def _check_for_required_arguments(cls: type(Params), kwargs: dict) -> None:
    """
    Make sure the arguments required by Params class or subclass exist in the kwargs dictionary
    used to instantiate the class
    """
    required_but_not_provided = []
    for k, v in cls.__dict__.items():
        if k.startswith('_') or not isinstance(v, BaseDescriptor):
            continue
        elif getattr(v, 'required', None) and not k in kwargs:
            required_but_not_provided.append(k)
    if required_but_not_provided != []:
        raise ValueError(f'{required_but_not_provided} are required arguments to instantiate '
                         f'{cls.__name__}')


def to_json_serializable_dict(params: Params, include_defaults: bool = True) -> dict:
    """
    Convert Params class to a json serializable dictionary
    """
    retval = {}
    for k, v in type(params).__dict__.items():
        if not k.startswith('_') and (include_defaults or k in params.__dict__):
            if isinstance(v, BaseDescriptor):
                retval[k] = v.to_json(params)
            elif isinstance(v, Params):
                retval[k] = to_json_serializable_dict(v, include_defaults)
    retval['_type'] = params.__class__.__name__
    retval['_module'] = params.__class__.__module__
    return retval


def from_json_serializable_dict(dictionary: dict) -> Params:
    """
    Convert from a json serializable dictionary to a Params class
    """
    temp_dictionary = copy.deepcopy(dictionary)
    try:
        cls_type = temp_dictionary.pop('_type')
        module_name = temp_dictionary.pop('_module')
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_type)
    except KeyError:
        raise ValueError('Dictionary Format Not Recognized!')
    # handle case with nested params
    unraveled_dictionary = {}
    for k, v in type(cls).__dict__.items():
        if not k.startswith('_'):
            if isinstance(v, BaseDescriptor):
                unraveled_dictionary[k] = temp_dictionary[k]
            elif isinstance(v, Params):
                unraveled_dictionary[k] = from_json_serializable_dict(temp_dictionary[k])
    return cls(**unraveled_dictionary)


# TODO: write functions to parse bringup params from a yaml and write them to a yaml
def to_yaml_file(params: Params, filename: str, include_defaults: bool=False):
    """
    Dump to yaml
    """
    with open(filename, 'w') as f:
        yaml.dump(to_json_serializable_dict(params, include_defaults=include_defaults), f)

def from_yaml_file(filename: str) -> Params:
    """
    Parse from yaml
    """
    with open(filename, 'r') as f:
        params = from_json_serializable_dict(yaml.full_load(f))
    return params


def merge_param_classes(params_cls_list = List[type(Params)],
                        merge_positional_params: bool = True) -> type(Params):
    """
    Merge multiple Params classes into a single merged params class and return the merged class
    """
    class MergedParams(Params):
        __doc__ = f'A Combination of {len(params_cls_list)} Params Classes:\n'

    for params_cls in params_cls_list:
        for k, v in params_cls.__dict__.items():
            if not k.startswith('_'):
                if hasattr(MergedParams, k):
                    raise ValueError(f'Unable to merge classes {params_cls_list} due to conflicting'
                                     f'param: {k}')
                setattr(MergedParams, k, v)
        MergedParams.__doc__ += f'\n\t {params_cls.__name__} - {params_cls.__doc__}'

    # resolve positional arguments:
    if merge_positional_params:
        if not sum([getattr(p, 'positional', False) for p in MergedParams.__dict__.values()]) > 1:
            return MergedParams
        positionals = {k:v for k, v in MergedParams.__dict__.items()
                       if getattr(v, 'positional', False)}
        # Just parse all positionals as a ListParam - we'll match them back up later in
        # _parse_positional_arguments
        positional_param = ListParam(help=', '.join(v.help for v in positionals.values()),
                                     positional=True, required=True, nargs='+')
        setattr(MergedParams, 'positionals', positional_param)
        for k in positionals.keys():
            delattr(MergedParams, k)
    return MergedParams


def _get_param_type(param) -> type:
    """
    Get the expected type of a param
    """
    argtype = match(param,
                    GeomspaceParam, lambda x: float,
                    ListParam, lambda x: param.subtype,
                    SetParam, lambda x: param.subtype,
                    EnumParam, lambda x: str,
                    IntParam, lambda x: int,
                    FloatParam, lambda x: float,
                    BoolParam, lambda x: None,
                    StringParam, lambda x: str,
                    BaseDescriptor, lambda x: type(getattr(param, 'default', None)))
    return argtype


def _convert_type_to_regex(argtype: type) -> str:
    """
    Source of truth for getting regex strings to match different types
    """
    return match(argtype,
                 type(int), lambda x: r'\b[\+-]?\d+\b',
                 type(float), lambda x: r'\b[\+-]?\d*\.\d*([eE][\+-]?\d+)?\b',
                 type(str), lambda x: r'\b.*\b')


def _parse_positional_arguments(list_of_positionals: List[str],
                                positional_params: Dict[str, BaseDescriptor]) -> dict:
    """
    If params have been merged together, their positional arguments will have been merged as well
    into one single ListParam. This function's job is to pair the positionals with their respective
    unmerged param classes.
    """
    if list_of_positionals is None and positional_params != {}:
        raise ValueError('No positional arguments were returned from the parser')
    elif list_of_positionals is None and positional_params == {}:
        return {}
    # first loop through combs for params that have choices specified
    copied_pos_params = copy.deepcopy(positional_params)
    copied_pos_list = copy.deepcopy(list_of_positionals)
    correctly_matched_params = {}
    params_to_delete = []
    for name, param in copied_pos_params.items():
        positionals_str = ' '.join(copied_pos_list)
        choices = getattr(param, 'choices', None)
        if choices is not None:
            nargs = str(getattr(param, 'nargs', 1))
            # convert to raw string to handle escape characters
            matched = re.findall(r'|'.join(fr'{c}' for c in choices), positionals_str)
            if matched == []:
                raise ValueError(f'Could not match any of the choices: {choices} in '
                                 f'positionals: {list_of_positionals} for param: {name}')
            if not nargs in ['+', '*', '?']:
                # only take the first nargs arguments
                matched = matched[:int(nargs)]
            # convert to expected type
            if getattr(param, 'nargs', None) is not None:
                correctly_matched_params[name] = [_get_param_type(param)(p) for p in matched]
            else:
                correctly_matched_params[name] = _get_param_type(param)(matched[0])
            # remove matches
            for p in matched:
                copied_pos_list.remove(p)
            params_to_delete.append(name)
    # do garbage collection at the end to avoid mutating the list we're iterating through
    for n in params_to_delete:
        del copied_pos_params[n]

    # second loop through matches to type - for nargs +, *, and ?, the behavior is greedy
    params_to_delete = []
    # first sort unmatched params by least to most restrictive type (int, float, str)
    def _type_sorter(p) -> int:
        argtype = _get_param_type(p[1])
        return match(argtype, type(int), 1, type(float), 2, type(str), 3, _, 4)

    for name, param in sorted(copied_pos_params.items(), key=_type_sorter):
        positionals_str = ' '.join(copied_pos_list)
        argtype = _get_param_type(param)
        pattern = _convert_type_to_regex(argtype)
        matched = re.findall(pattern, positionals_str)
        if matched == []:
            raise ValueError(f'Unable to find {argtype} in {list_of_positionals} for param: {name}')
        nargs = str(getattr(param, 'nargs', 1))
        if not nargs in ['+', '*', '?']:
            matched = matched[:int(nargs)]
        # convert to expected type
        if getattr(param, 'nargs', None) is not None:
            correctly_matched_params[name] = [_get_param_type(param)(p) for p in matched]
        else:
            correctly_matched_params[name] = _get_param_type(param)(matched[0])
        # remove matches
        for p in matched:
            copied_pos_list.remove(p)
        params_to_delete.append(name)
    # do garbage collection at the end to avoid mutating the list we're iterating through
    for n in params_to_delete:
        del copied_pos_params[n]

    # No third Loop
    assert copied_pos_params == {} and copied_pos_list == [], 'Unable to parse positional arguments'
    return correctly_matched_params


def _add_param_to_parser(name: str, param: BaseDescriptor, parser: ArgumentParser) -> None:
    """
    Function to add a Param like IntParam, FloatParam, etc. called <name> to a parser
    """
    argtype = _get_param_type(param)
    if argtype == type(None):
        raise NotImplementedError(f'Argparse type not implemented '
                                  f'for {param.__class__.__name__} and default not specifed')
    positional = getattr(param, 'positional', False)
    argname = name if positional else '--' + name
    required = True if getattr(param, 'required', False) else None
    default = param.default if required is None else None
    unit = getattr(param, 'unit', None)
    if param.help != SUPPRESS:
        help = f'{param.help} [default: {default} {unit}]' \
            if unit is not None else f'{param.help} [default: {default}]'
    else:
        help = param.help
    if not required and positional:
        # TODO: use nargs='*' or nargs='?' to support not-required positional arguments
        raise ValueError('Not-required positional arguments are currently not supported')
    elif positional:
        # positional arguments are required by default, and argparse complains if you specify
        # required = True
        required = None
    action = match(param,
                   BoolParam, lambda p: 'store_true' if not p.default else 'store_false',
                   BaseDescriptor, lambda p: None)
    nargs = getattr(param, 'nargs', None)
    assert not (action is not None and nargs is not None)
    choices = match(param,
                    EnumParam, lambda x: list(x.cls.__members__.keys()),
                    BaseDescriptor, lambda x: getattr(x, 'choices', None))
    kwargs = dict(action=action, nargs=nargs, default=default,
                  type=argtype, required=required, help=help, choices=choices)
    # we delete all kwargs that are None to avoid hitting annoying action class initializations
    # such as when action is store_true and 'nargs' is in kwargs
    for kw in list(kwargs.keys()):
        if kwargs[kw] is None:
            del kwargs[kw]
    parser.add_argument(argname, **kwargs)


def _expand_multi_arg_param(name: str, param: BaseDescriptor) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Expand a parameter like GeomspaceParam or ArangeParam into seperate IntParams and FloatParams to
    parse as '--start X --stop X --num X' or '--start X --stop X --step X', etc.
    """
    new_arg_names = match(param,
                          GeomspaceParam, ['start', 'stop', 'num'],
                          ArangeParam, ['start', 'stop', 'step'])
    if getattr(param, 'positional', False):
        raise ValueError(f'Cannot expand positional {param.__class__.__name__} to {new_arg_names}')
    expanded_types = match(param,
                           GeomspaceParam, [FloatParam, FloatParam, IntParam],
                           ArangeParam, [FloatParam, FloatParam, FloatParam])
    unit = getattr(param, 'unit', None)
    expanded_units = match(param,
                           GeomspaceParam,
                           lambda p: [unit, unit, None],
                           ArangeParam,
                           lambda p: [unit, unit, unit])
    defaults = getattr(param, 'default', [None, None, None])
    if defaults is None:
        defaults = [None, None, None]
    choices = getattr(param, 'choices', None)
    expanded_choices = [[], [], []]
    if choices is not None:
        # transpose the choices for easy splitting but preserve type
        for c in choices:
            expanded_choices[0].append(c[0])
            expanded_choices[1].append(c[1])
            expanded_choices[2].append(c[2])
    expanded_params = []
    for i, (default, argtype, u) in enumerate(zip(defaults, expanded_types, expanded_units)):
        new_param = argtype(help=param.help + ' (expanded into three arguments)',
                            default=default,
                            required=getattr(param,'required', False),
                            choices=expanded_choices[i] if choices is not None else None,
                            unit=u)
        expanded_params.append((new_arg_names[i], new_param))
    # we add the param we're expanding to the list as well to make sure the user does not pass it
    # Otherwise, it will get parsed silently, and the user will wonder what's happening
    copied_param = copy.deepcopy(param)
    setattr(copied_param, 'required', False)
    setattr(copied_param, 'help', SUPPRESS)
    setattr(copied_param, 'default', None)
    expanded_params.append((name, copied_param))
    return tuple(expanded_params)


def _extract_expanded_param(parsed_values: dict,
                            name: str,
                            param: BaseDescriptor) -> Optional[List]:
    """
    Convert [start, stop, num] or [start, stop, step], etc. from expanded form back into a list to
    be easily fed into the un-expanded param
    """
    old_arg_names = match(param,
                          GeomspaceParam, ['start', 'stop', 'num'],
                          ArangeParam, ['start', 'stop', 'step'])
    assert parsed_values.get(name) is None, \
        f'{name} was expanded. Please provide {old_arg_names} instead'
    start_stop_x_list = [parsed_values[n] for n in old_arg_names]
    if all([x is None for x in start_stop_x_list]):
        return None
    return start_stop_x_list


def _flatten_cls_params(cls: type(Params)) -> Dict:
    """
    Extract params from a Params class - Behavior is as follows:

    1. Params with names starting in _ will be ignored
    2. Derived params will be ignored
    3. If the class contains nested params classes, those will be flattened. Params with the same
        name in the encompassing class will overwrite the nested ones if overwrite is specified
        Otherwise, an error will be thrown.
    """
    flattend_params = {}
    # loop through and all the nested params first
    already_flat_params = {}
    for name, param in vars(cls).items():
        if name.startswith('_'):
            continue
        elif isinstance(param, BaseDescriptor):
            if name in already_flat_params:
                raise KeyError(f'Unable to flatten {cls.__name__} - conflict with param: {name}')
            already_flat_params[name] = param
        elif isinstance(param, Params):
            for n, p in _flatten_cls_params(type(param)).items():
                if n in flattend_params:
                    raise KeyError(f'Unable to flatten {cls.__name__} - conflict with param: {n}')
                flattend_params[n] = p
        elif isinstance(param, property):
            continue
        else:
            raise ValueError(f'Param: {name} of type {type(param)} is not recognized!')

    for name, param in already_flat_params.items():
        if name in flattend_params and not getattr(param, 'override', False):
            raise KeyError(f'Unable to flatten {cls.__name__} - conflict with param: {name} - use'
                           f' kwarg override=True to override nested params')
        flattend_params[name] = param
    return flattend_params


def to_argparse(cls: type(Params)) -> ArgumentParser:
    """
    Convert a Params class or subclass to an argparse argument parser.
    """
    parser = ArgumentParser(description=cls.__doc__)

    for name, param in _flatten_cls_params(cls).items():
        if getattr(param, 'expand', False):
            for (n, p) in _expand_multi_arg_param(name, param):
                _add_param_to_parser(n, p, parser)
        else:
            _add_param_to_parser(name, param, parser)

    assert sum(getattr(p, 'nargs', '') not in ['+', '*', '?'] for p in vars(cls).values()
               if getattr(p, 'positional', False)) <= 1, \
        'Behavior is undefined for multiple positional arguments with nargs=+|*|?'
    assert sum(getattr(p, 'expand', False) for p in vars(cls).values()) <= 1, \
        'Cannot expand multiple params into (start, stop, ...) for use in the same parser'
    return parser


def from_parsed_args(cls_list: Tuple[type(Params)], params_namespace: Namespace) -> Tuple:
    """
    Convert a list of params classes and an argparse parsed namespace to a list of class instances
    """
    params = vars(params_namespace)

    # handle positional arguments
    positional_params = {}
    for cls in cls_list:
        positional_params.update({k: v for k, v in cls.__dict__.items()
                                  if getattr(v, 'positional', False)})
    # now we must match the positional params to the list of positional arguments we got back
    matched_pos_args = _parse_positional_arguments(params.get('positionals', None),
                                                   positional_params)
    params.update(matched_pos_args)

    # handle expanded params
    expanded_params = {}
    for cls in cls_list:
        expanded_params.update({k: _extract_expanded_param(params, k, v)
                                for k, v in vars(cls).items() if getattr(v, 'expand', False)})
    params.update(expanded_params)

    # actually construct the params classes

    params_instance_list = []
    for cls in cls_list:
        cls_specific_params = {k: params[k] for k, v in cls.__dict__.items()
                               if not k.startswith('_') and
                               k in params and
                               isinstance(v, BaseDescriptor)}
        params_instance_list.append(cls(**cls_specific_params))
    return tuple(params_instance_list)


def create_parser_and_parser_args(*cls, throw_on_unknown: bool=False
                                  ) -> Union[Tuple[Params], Params]:
    """
    Outside interface for creating a parser from multiple Params classes and parsing arguments
    """
    if len(cls) > 1:
        parser = to_argparse(merge_param_classes(cls))
    else:
        parser = to_argparse(cls[0])
    args, argv = parser.parse_known_args()
    if argv != [] and throw_on_unknown:
        raise ValueError(f'Unknown arguments: {argv}')
    return from_parsed_args(cls, args) if len(cls) > 1 else from_parsed_args(cls, args)[0]
