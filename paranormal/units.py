from typing import Optional

###################
# Unit Conversion #
###################


UNIT_CONVERSION_TABLE = {"GHz" : 1.0e9,
                         "MHz": 1.0e6,
                         "ns": 1.0e-9,
                         "μs": 1.0e-6,
                         "us": 1.0e-6,
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


def unconvert_si_units(value, unit: Optional[str] = None):
    """
    Convert from si units back to "unit" units
    """
    if unit is None:
        return value
    try:
        return value / UNIT_CONVERSION_TABLE[unit]
    except KeyError:
        raise KeyError(f'Unit "{unit}" not in conversion table!')
    except TypeError:
        raise TypeError(f'Unit "{unit}" not compatible with value {value} of type {type(value)}')
    