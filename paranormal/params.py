from abc import ABC
from enum import Enum, EnumMeta
from typing import Iterable, List, Optional, Set, Tuple, Union
import warnings

import numpy as np

from paranormal.units import convert_to_si_units


__all__ = ['BaseDescriptor', 'BoolParam', 'FloatParam', 'IntParam', 'StringParam', 'ListParam',
           'SetParam', 'ArangeParam', 'EnumParam', 'GeomspaceParam', 'SpanArangeParam',
           'LinspaceParam']


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

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

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
        if default not in cls and default is not None:
            raise ValueError(f'Default {default} is not a member of {cls}')
        self.cls = cls
        self.default = default
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
                  f'{nargs} without any None values. Default behavior will not work')
    return False


class NumpyFunctionParam(BaseDescriptor):
    nargs = 3
    def __init__(self, *,
                 help: str,
                 default: Optional[Union[List, Tuple]] = None,
                 required: Optional[bool] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        if default is not None and required:
            raise ValueError('Default cannot be specified if required is True!')
        self.default = default
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
        return convert_to_si_units(self._numpy_function(v), self.unit)

    def __set__(self, instance, value):
        _check_numpy_fn_param_value(self.name, value, self.nargs)
        instance.__dict__[self.name] = value

    def _numpy_function(self, value: Iterable) -> np.ndarray:
        raise NotImplementedError()

    def to_json(self, instance, include_default: bool = True):
        v = instance.__dict__.get(self.name, self.default)
        if _check_numpy_fn_param_value(self.name, v, self.nargs):
            return super(NumpyFunctionParam, self).to_json(instance, include_default)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        else:
            return v


class GeomspaceParam(NumpyFunctionParam):
    def __set__(self, instance, value):
        if _check_numpy_fn_param_value(self.name, value, self.nargs):
            assert value[0] != 0
        super(GeomspaceParam, self).__set__(instance, value)

    def _numpy_function(self, value: Iterable):
        return np.geomspace(*value)


class ArangeParam(NumpyFunctionParam):
    def _numpy_function(self, value: Iterable):
        return np.arange(*value)


class SpanArangeParam(NumpyFunctionParam):
    def _numpy_function(self, value: Iterable):
        center, width, step = value[0], value[1], value[2]
        return np.arange(center - 0.5 * width, center + 0.5 * width, step)


class LinspaceParam(NumpyFunctionParam):
    def _numpy_function(self, value: Iterable):
        return np.linspace(value[0], value[1], int(value[2]))
