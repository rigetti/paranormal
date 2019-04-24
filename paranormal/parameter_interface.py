from argparse import ArgumentParser, Namespace, SUPPRESS
import copy
from collections import defaultdict, Mapping
import importlib
import re
from typing import Dict, List, Optional, Tuple, Union
import yaml

from pampy import match

from paranormal.params import *


__all__ = ['Params', 'to_json_serializable_dict', 'from_json_serializable_dict', 'to_yaml_file',
           'from_yaml_file', 'create_parser_and_parse_args', 'to_argparse', 'from_parsed_args']


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
                raise KeyError(f'{cls.__name__} does not take {key} as an argument')
            setattr(self, key, value)
        _check_for_required_arguments(cls, kwargs)
        _ensure_properties_are_working(self)

    def __iter__(self):
        valid_keys = [k for k in self.__class__.__dict__.keys() if not k.startswith('_')]
        for key in valid_keys:
            yield key

    def __len__(self):
        return len(list(self.__iter__()))

    def __getitem__(self, item):
        return getattr(self, item)

    def __eq__(self, other):
        return to_json_serializable_dict(self) == to_json_serializable_dict(other)


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
        raise KeyError(f'{required_but_not_provided} are required arguments to instantiate '
                       f'{cls.__name__}')


def _ensure_properties_are_working(params: Params) -> None:
    """
    Evaluate the properties within a class to make sure they work based on the attributes
    """
    for k, v in params.items():
        if isinstance(v, property):
            getattr(params, k)


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
    for k, v in cls.__dict__.items():
        if not k.startswith('_'):
            if isinstance(v, BaseDescriptor) and k in temp_dictionary:
                unraveled_dictionary[k] = temp_dictionary[k]
            elif isinstance(v, Params):
                unraveled_dictionary[k] = from_json_serializable_dict(temp_dictionary[k])
    return cls(**unraveled_dictionary)


def to_yaml_file(params: Params, filename: str, include_defaults: bool=False):
    """
    Dump to yaml
    """
    with open(filename, 'w') as f:
        yaml.dump(to_json_serializable_dict(params, include_defaults=include_defaults), stream=f)

def from_yaml_file(filename: str) -> Params:
    """
    Parse from yaml
    """
    with open(filename, 'r') as f:
        params = from_json_serializable_dict(yaml.full_load(f))
    return params


def _merge_positional_params(params_list: List[Tuple[str, BaseDescriptor]]
                             ) -> Tuple[List[str], Optional[BaseDescriptor]]:
    """
    Merge positional params into a single list param and return a list of param names that were
    merged
    """
    if not sum([getattr(p, 'positional', False) for (_, p) in params_list]) > 1:
        return [], None
    positionals = {k: v for (k, v) in params_list
                   if getattr(v, 'positional', False)}
    # Just parse all positionals as a ListParam - we'll match them back up later in
    # _parse_positional_arguments
    positional_param = ListParam(help=', '.join(v.help for v in positionals.values()
                                                if v.help != SUPPRESS),
                                 positional=True, required=True, nargs='+')
    return list(positionals.keys()), positional_param


def _merge_param_classes(params_cls_list = List[type(Params)],
                         merge_positional_params: bool = True) -> type(Params):
    """
    Merge multiple Params classes into a single merged params class and return the merged class
    """
    if len(params_cls_list) == 1:
        return params_cls_list[0]

    class MergedParams(Params):
        __doc__ = f'A Combination of {len(params_cls_list)} Params Classes:\n'

    for params_cls in params_cls_list:
        for k, v in params_cls.__dict__.items():
            if not k.startswith('_'):
                if hasattr(MergedParams, k):
                    raise ValueError(f'Unable to merge classes {params_cls_list} due to conflicting'
                                     f'param: {k}')
                setattr(MergedParams, k, v)
                v.__set_name__(MergedParams, k)
        MergedParams.__doc__ += f'\n\t {params_cls.__name__} - {params_cls.__doc__}'

    # resolve positional arguments:
    if merge_positional_params:
        params_to_delete, positional_param = _merge_positional_params(
            [(k, v) for k, v in MergedParams.__dict__.items() if not k.startswith('_')])
        if positional_param is None and params_to_delete == []:
            return MergedParams
        setattr(MergedParams, 'positionals', positional_param)
        positional_param.__set_name__(MergedParams, 'positionals')
        for k in params_to_delete:
            delattr(MergedParams, k)

    return MergedParams


def _get_param_type(param) -> type:
    """
    Get the expected type of a param
    """
    argtype = match(param,
                    LinspaceParam, lambda x: float,
                    ArangeParam, lambda x: float,
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
    regex_patterns = {int : r'\b[\+-]?(?<![\.\d])\d+(?!\.\d)\b',
                      float : r'[-\+]?(?:\d+(?<!\.)\.?(?!\.)\d*|\.?\d+)(?:[eE][-\+]?\d+)?',
                      str : r'\b.*\b'}
    return regex_patterns[argtype]


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
        choices = getattr(param, 'choices', None)
        if choices is not None:
            nargs = str(getattr(param, 'nargs', 1))
            # convert to raw string to handle escape characters
            matched = [p for p in copied_pos_list
                       if re.match(r'|'.join(fr'{c}' for c in choices), p)]
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
        order = defaultdict(lambda x: 4)
        order.update({int: 1, float: 2, str: 3})
        return order[argtype]

    for name, param in sorted(copied_pos_params.items(), key=_type_sorter):
        argtype = _get_param_type(param)
        pattern = _convert_type_to_regex(argtype)
        matched = [p for p in copied_pos_list if re.match(pattern, p)]
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
    if (getattr(param, 'prefix', '') != '' and not getattr(param, 'expand', False)):
        raise ValueError(f'Failure with param {name}. Cannot add a prefix to a class without the'
                         f' expand kwarg set to True')
    argname = name if positional else '--' + name
    required = True if getattr(param, 'required', False) else None
    default = param.default if required is None else None
    unit = getattr(param, 'unit', None)

    # format help nicely if default is specified and suppress is not set
    if positional or param.help == SUPPRESS:
        help = param.help
    else:
        help = f'{param.help} [default: {default} {unit}]' \
            if unit is not None else f'{param.help} [default: {default}]'
    if not required and positional:
        # TODO: use nargs='*' or nargs='?' to support not-required positional arguments
        raise ValueError('Not-required positional arguments are currently not supported')
    elif positional:
        # positional arguments are required by default, and argparse complains if you specify
        # required = True
        required = None
        default = None
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


def _expand_param_name(param: BaseDescriptor) -> List[str]:
    """
    Get expanded param names
    """
    if not getattr(param, 'expand', False):
        raise ValueError('Cannot expand param that does not have the expand kwarg')
    new_arg_names = match(param,
                          SpanArangeParam, ['center', 'width', 'step'],
                          GeomspaceParam, ['start', 'stop', 'num'],
                          ArangeParam, ['start', 'stop', 'step'],
                          LinspaceParam, ['start', 'stop', 'num'])
    prefix = getattr(param, 'prefix', '')
    new_arg_names = [prefix + n for n in new_arg_names]
    return new_arg_names


def _expand_multi_arg_param(name: str, param: BaseDescriptor) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Expand a parameter like GeomspaceParam or ArangeParam into seperate IntParams and FloatParams to
    parse as '--start X --stop X --num X' or '--start X --stop X --step X', etc.
    """
    new_arg_names = _expand_param_name(param)
    if getattr(param, 'positional', False):
        raise ValueError(f'Cannot expand positional {param.__class__.__name__} to {new_arg_names}')
    expanded_types = match(param,
                           GeomspaceParam, [FloatParam, FloatParam, IntParam],
                           ArangeParam, [FloatParam, FloatParam, FloatParam],
                           LinspaceParam, [FloatParam, FloatParam, IntParam])
    unit = getattr(param, 'unit', None)
    expanded_units = match(param,
                           GeomspaceParam,
                           lambda p: [unit, unit, None],
                           ArangeParam,
                           lambda p: [unit, unit, unit],
                           LinspaceParam,
                           lambda p: [unit, unit, None])
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
    old_arg_names = _expand_param_name(param)
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
    4. Multiple positional params will be merged after flattening
    """
    flattened_params = {}
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
                if n in flattened_params:
                    raise KeyError(f'Unable to flatten {cls.__name__} - conflict with param: {n}')
                flattened_params[n] = p
        elif isinstance(param, property):
            continue
        else:
            raise ValueError(f'Param: {name} of type {type(param)} is not recognized!')

    for name, param in already_flat_params.items():
        if name in flattened_params and not getattr(param, 'override', False):
            raise KeyError(f'Unable to flatten {cls.__name__} - conflict with param: {name} - use'
                           f' kwarg override=True to override nested params')
        flattened_params[name] = param

    return flattened_params


def to_argparse(cls: type(Params), **kwargs) -> ArgumentParser:
    """
    Convert a Params class or subclass to an argparse argument parser.
    """
    parser = ArgumentParser(description=cls.__doc__, **kwargs)

    # first, we flatten the cls params (means flattening any nested Params classes
    flattened_params = _flatten_cls_params(cls)

    # merge any positional arguments
    params_to_delete, positional_param = _merge_positional_params(list(flattened_params.items()))
    for p in params_to_delete:
        del flattened_params[p]
    if positional_param is not None:
        flattened_params['positionals'] = positional_param

    # Check if multiple params are to be expanded. If so, we need a prefix on each of the expanded
    # names to avoid conflict
    if sum(getattr(p, 'expand', False) for p in flattened_params.values()) > 1:
        for p in flattened_params.values():
            assert not (getattr(p, 'expand', False) ^ (getattr(p, 'prefix', '') != ''))

    for name, param in flattened_params.items():
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


def _unflatten_params_cls(cls: type(Params), parsed_params: dict) -> Params:
    """
    Recursively construct a Params class or subclass (handles nested Params classes)
    """
    cls_specific_params = {}
    for k, v in cls.__dict__.items():
        if k.startswith('_'):
            continue
        elif isinstance(v, BaseDescriptor) and k in parsed_params:
            cls_specific_params[k] = parsed_params[k]
        elif isinstance(v, Params):
            cls_specific_params[k] = _unflatten_params_cls(type(v), parsed_params)
    return cls(**cls_specific_params)


def from_parsed_args(*cls_list, params_namespace: Namespace) -> Tuple:
    """
    Convert a list of params classes and an argparse parsed namespace to a list of class instances
    """
    params = vars(params_namespace)

    flattened_classes = [_flatten_cls_params(cls) for cls in cls_list]
    # handle positional arguments
    positional_params = {}
    for flattened_cls in flattened_classes:
        positional_params.update({k: v for k, v in flattened_cls.items()
                                  if getattr(v, 'positional', False)})
    if 'positionals' in params:
        parsed_positionals = params['positionals']
        # now we must match the positional params to the list of positional arguments we got back
        matched_pos_args = _parse_positional_arguments(parsed_positionals,
                                                       positional_params)
        params.update(matched_pos_args)

    # handle expanded params
    expanded_params = {}
    for flattened_cls in flattened_classes:
        expanded_params.update({k: _extract_expanded_param(params, k, v)
                                for k, v in flattened_cls.items() if getattr(v, 'expand', False)})
    params.update(expanded_params)

    # actually construct the params classes
    return tuple(_unflatten_params_cls(cls, params) for cls in cls_list)


def create_parser_and_parse_args(*cls,
                                 throw_on_unknown: bool = False,
                                 **kwargs) -> Union[Tuple[Params], Params]:
    """
    Outside interface for creating a parser from multiple Params classes and parsing arguments
    """
    parser = to_argparse(_merge_param_classes(cls), **kwargs)
    args, argv = parser.parse_known_args()
    if argv != [] and throw_on_unknown:
        raise ValueError(f'Unknown arguments: {argv}')
    return from_parsed_args(*cls, params_namespace=args) \
        if len(cls) > 1 else from_parsed_args(*cls, params_namespace=args)[0]
