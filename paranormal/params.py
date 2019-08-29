from abc import ABC
from enum import Enum, EnumMeta
from typing import Iterable, List, Optional, Set, Tuple
import warnings

import numpy as np

from paranormal.units import convert_to_si_units


__all__ = ['BaseDescriptor', 'BoolParam', 'FloatParam', 'IntParam', 'StringParam', 'ListParam',
           'SetParam', 'ArangeParam', 'EnumParam', 'GeomspaceParam', 'SpanArangeParam',
           'LinspaceParam', 'SpanLinspaceParam']


#################################
# All the Different Param Types #
#################################

class BaseDescriptor(ABC):
    """
    Abstract base type for overriding python descriptor behavior
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name, None)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def to_json(self, instance, include_default: bool = True):
        """
        Method to convert this parameter's value to something that's JSON serializable
        """
        return instance.__dict__.get(self.name,
                                     getattr(self, 'default', None) if include_default else None)


class BoolParam(BaseDescriptor):
    """
    A Boolean that overrides the python descriptor behavior
    """
    def __init__(self, *, help: str, default: bool, hide: bool = False, **kwargs):
        """
        :param help: A string that describes what this boolean is for
        :param default: The default value for this boolean
        :param hide: Whether or not to hide this parameter from the argument parser
        :param kwargs: Extra kwargs to modify behavior (not used)
        """
        if 'required' in kwargs or default is None:
            raise ValueError('BoolParam cannot be a required argument '
                             'and must have a default value')
        self.default = default
        self.help = help
        self.hide = hide
        super(BoolParam, self).__init__(**kwargs)

    def __get__(self, instance, owner) -> bool:
        return instance.__dict__.get(self.name, self.default)


class FloatParam(BaseDescriptor):
    """
    A Float that overrides the python descriptor behavior
    """
    def __init__(self, *,
                 help: str,
                 default: Optional[float] = None,
                 hide: bool = False,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        """
        :param help: A string that describes what this float is for
        :param default: The default value for this parameter
        :param hide: Hide this parameter from the argument parser
        :param required: Whether or not this parameter is required when parsing using an
            argument parser
        :param unit: A unit for this parameter (check paranormal.units for possible units)
        :param kwargs: Extra kwargs to modify behavior:
            'choices': list or tuple of all possible choices for this parameter's value - only
                relevant for the argument parser (consider an EnumParam for something more strict)
            'positional' Whether or not to parse this parameter as a positional
        """
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.hide = hide
        self.required = required
        self.help = help
        self.unit = unit
        super(FloatParam, self).__init__(**kwargs)

    def __get__(self, instance, owner) -> Optional[float]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        return instance.__dict__.get(self.name, self.default)


class IntParam(BaseDescriptor):
    """
    An Int that overrides the python descriptor behavior
    """
    def __init__(self, *,
                 help: str,
                 default: Optional[int] = None,
                 hide: bool = False,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        """
        :param help: A string that describes what this int is for
        :param default: The default value for this parameter
        :param hide: Hide this parameter from the argument parser
        :param required: Whether or not this parameter is required when parsing using an
            argument parser
        :param unit: A unit for this parameter (check paranormal.units for possible units)
        :param kwargs: Extra kwargs to modify behavior:
            'choices': list or tuple of all possible choices for this parameter's value - only
                relevant for the argument parser (consider an EnumParam for something more strict)
            'positional' Whether or not to parse this parameter as a positional
        """
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.hide = hide
        self.required = required
        self.help = help
        self.unit = unit
        super(IntParam, self).__init__(**kwargs)

    def __get__(self, instance, owner) -> Optional[int]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        return instance.__dict__.get(self.name, self.default)


class StringParam(BaseDescriptor):
    """
    A string that overrides the python descriptor behavior
    """
    def __init__(self, *,
                 help: str,
                 default: Optional[str] = None,
                 hide: bool = False,
                 required: Optional[bool] = None,
                 **kwargs):
        """
        :param help: A string that describes what this int is for
        :param default: The default value for this parameter
        :param hide: Hide this parameter from the argument parser
        :param required: Whether or not this parameter is required when parsing using an
            argument parser
        :param unit: A unit for this parameter (check paranormal.units for possible units)
        :param kwargs: Extra kwargs to modify behavior:
            'choices': list or tuple of all possible choices for this parameter's value - only
                relevant for the argument parser (consider an EnumParam for something more strict)
            'positional' Whether or not to parse this parameter as a positional
        """
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.hide = hide
        self.required = required
        self.help = help
        super(StringParam, self).__init__(**kwargs)

    def __get__(self, instance, owner) -> Optional[str]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        if instance.__dict__.get(self.name, self.default) is None:
            return None
        return str(instance.__dict__.get(self.name, self.default))


class ListParam(BaseDescriptor):
    """
    A list that overrides the python descriptor behavior
    """
    def __init__(self, *,
                 help: str,
                 subtype: Optional[type] = None,
                 default: Optional[List] = None,
                 hide: bool = False,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        """
        :param help: A string that describes what this list is for
        :param subtype: The type of objects that make up this list (specify if you want to parse
            using an argument parser)
        :param default: The default value for this parameter
        :param hide: Hide this parameter from the argument parser
        :param required: Whether or not this parameter is required when parsing using an
            argument parser
        :param unit: A unit for this parameter (check paranormal.units for possible units)
        :param kwargs: Extra kwargs to modify behavior:
            'choices': list or tuple of all possible choices for this items in the list - only
                relevant for the argument parser (consider an EnumParam for something more strict)
            'positional' Whether or not to parse this parameter as a positional
            'nargs': How many arguments this parameter should take when parsing from the command
                line
        """
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.hide = hide
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
        return list(instance.__dict__.get(self.name, self.default))


class SetParam(BaseDescriptor):
    """
    A set that overrides python descriptor behavior
    """
    def __init__(self, *,
                 help: str,
                 subtype: type = str,
                 default: Optional[Set] = None,
                 hide: bool = False,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        """
        :param help: A string that describes what this set is for
        :param subtype: The type of objects that make up this list (specify if you want to parse
            using an argument parser)
        :param default: The default value for this parameter
        :param hide: Hide this parameter from the argument parser
        :param required: Whether or not this parameter is required when parsing using an
            argument parser
        :param unit: A unit for this parameter (check paranormal.units for possible units)
        :param kwargs: Extra kwargs to modify behavior:
            'choices': list or tuple of all possible choices for this items in the list - only
                relevant for the argument parser (consider an EnumParam for something more strict)
            'positional' Whether or not to parse this parameter as a positional
            'nargs': How many arguments this parameter should take when parsing from the command
                line
        """
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
        self.hide = hide
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
        return set(instance.__dict__.get(self.name, self.default))


class EnumParam(BaseDescriptor):
    """
    An enum that overrides python descriptor behavior
    """
    def __init__(self, *,
                 help: str,
                 cls: EnumMeta,
                 default: Optional[Enum] = None,
                 hide: bool = False,
                 required: Optional[bool] = None,
                 **kwargs):
        """
        :param help: A string that describes what this enum is for
        :param cls: The enum class this parameter represents
        :param default: The default value for this parameter
        :param hide: Hide this parameter from the argument parser
        :param required: Whether or not this parameter is required when parsing using an
            argument parser
        :param kwargs: Extra kwargs to modify behavior:
            'positional' Whether or not to parse this parameter as a positional
        """
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        if default not in cls and default is not None:
            raise ValueError(f'Default {default} is not a member of {cls}')
        self.cls = cls
        self.default = default
        self.hide = hide
        self.required = required
        self.help = help
        super(EnumParam, self).__init__(**kwargs)

    def __get__(self, instance, owner) -> Optional[Enum]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        if value is None or value in self.cls:
            instance.__dict__[self.name] = value
        elif value in self.cls.__members__:
            instance.__dict__[self.name] = self.cls[value]
        else:
            raise KeyError(f'{value} is not a valid member of {self.cls.__name__}')

    def to_json(self, instance, include_default: bool = True):
        default = getattr(self, 'default', None) if include_default else None
        tmp = instance.__dict__.get(self.name, default)
        return tmp.name if tmp is not None else None


###############################
# Numpy function based params #
###############################


def _check_numpy_fn_param_value(name: str, value: Iterable, nargs: int) -> bool:
    """
    Check whether a value is fit for use in a three argument numpy function
    """
    if (isinstance(value, (tuple, list)) and len(value) == nargs and
            not any([i is None for i in value])):
        return True
    warnings.warn(f'Param {name} does not match format required (List or Tuple of length '
                  f'{nargs} without any None values. Default behavior will not work.')
    return False


class NumpyFunctionParam(BaseDescriptor):
    """
    A parameter that is set with 3 arguments and returns the result of a numpy function called
    on those arguments if those arguments satisfy _check_numpy_fn_param_value
    """
    nargs = 3

    def __init__(self, *,
                 help: str,
                 default: Optional[Tuple] = None,
                 hide: bool = False,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        """
        :param help: A string that describes what this parameter is for
        :param subtype: The type of objects that make up this list (specify if you want to parse
            using an argument parser)
        :param default: The default value for this parameter
        :param hide: Hide this parameter from the argument parser
        :param required: Whether or not this parameter is required when parsing using an
            argument parser
        :param unit: A unit for this parameter (check paranormal.units for possible units)
        :param kwargs: Extra kwargs to modify behavior:
            'choices': list or tuple of all possible choices for this parameter's value - only
                relevant for the argument parser (consider an EnumParam for something more strict)
            'expand': Whether or not to expand this parameter into sub arguments when argument
                parsing
            'positional' Whether or not to parse this parameter as a positional
            'nargs': How many arguments this parameter should take when parsing from the command
                line - overwrites the default of 3
        """
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        if not isinstance(default, tuple) and default is not None:
            raise ValueError('Default must be either a tuple or None')
        self.default = default
        self.hide = hide
        self.required = required
        self.help = help
        self.unit = unit
        super(NumpyFunctionParam, self).__init__(**kwargs)

    def __get__(self, instance, owner) -> Optional[np.ndarray]:
        if self.required and self.name not in instance.__dict__:
            raise ValueError(f'{self.name} is a required argument and must be set first!')
        v = instance.__dict__.get(self.name, self.default)
        if not _check_numpy_fn_param_value(self.name, v, self.nargs):
            return v
        return self._numpy_function(v)

    def __set__(self, instance, value):
        _check_numpy_fn_param_value(self.name, value, self.nargs)
        instance.__dict__[self.name] = value

    def _numpy_function(self, value: Iterable) -> np.ndarray:
        raise NotImplementedError()

    def to_json(self, instance, include_default: bool = True):
        """
        Convert this parameter to a json serializable value
        """
        v = instance.__dict__.get(self.name, self.default)
        if _check_numpy_fn_param_value(self.name, v, self.nargs):
            return super(NumpyFunctionParam, self).to_json(instance, include_default)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        else:
            return v


class GeomspaceParam(NumpyFunctionParam):
    """
    A parameter that is set with 3 arguments (start, stop, num) and returns the geomspace function
    called on those arguments if those arguments satisfy _check_numpy_fn_param_value
    """
    def __set__(self, instance, value):
        if _check_numpy_fn_param_value(self.name, value, self.nargs):
            assert value[0] != 0
        super(GeomspaceParam, self).__set__(instance, value)

    def _numpy_function(self, value: Iterable):
        return np.geomspace(*value)


class ArangeParam(NumpyFunctionParam):
    """
    A parameter that is set with 3 arguments (start, stop, step) and returns the arange function
    called on those arguments if those arguments satisfy _check_numpy_fn_param_value
    """
    def _numpy_function(self, value: Iterable):
        return np.arange(*value)


class SpanArangeParam(NumpyFunctionParam):
    """
    A parameter that is set with 3 arguments (center, width, step) and returns the arange function
    called on (center - 0.5 * width, center + 0.5 * width, step) if those arguments satisfy
    _check_numpy_fn_param_value
    """
    def _numpy_function(self, value: Iterable):
        center, width, step = value[0], value[1], value[2]
        return np.arange(center - 0.5 * width, center + 0.5 * width, step)


class LinspaceParam(NumpyFunctionParam):
    """
    A parameter that is set with 3 arguments (start, stop, num) and returns the linspace function
    called on those arguments if those arguments satisfy _check_numpy_fn_param_value
    """
    def _numpy_function(self, value: Iterable):
        return np.linspace(value[0], value[1], int(value[2]))


class SpanLinspaceParam(NumpyFunctionParam):
    """
    A parameter that is set with 3 arguments (center, width, num) and returns the linspace function
    called on (center - 0.5 * width, center + 0.5 * width, num) if those arguments satisfy
    _check_numpy_fn_param_value
    """
    def _numpy_function(self, value: Iterable):
        center, width, num = value[0], value[1], value[2]
        return np.linspace(center - 0.5 * width, center + 0.5 * width, int(num))
