# paranormal

A declarative, parameter-parsing library that provides multiple parsing interfaces (YAML, command line, and JSON) for loading parameters.

## Python Install:

Just install from PyPi using `pip install paranormal`.

## Running unit tests

Unit tests can be executed by running `pytest` from top folder of the repository.

## Using the Library

The following code samples show how this library is meant to be used.

### Subclass `Params`

```python
from paranormal.parameter_interface import *
from paranormal.params import *

class FrequencySweep(Params):
    """
    A frequency sweep measurement
    """
    freqs = SpanArangeParam(help='A list of frequencies to scan as [center, width, step]',
                            default=[1, 2, 0.1], unit='GHz')
    power = FloatParam(help='Power to transmit', default=-20, unit='dBm')
    pulse_samples = IntParam(help='Samples in the pulse', default=100)
    averages = IntParam(help='Number of sweeps to average over', default=10)
    is_test = BoolParam(help='Is this just a test', default=False)

```


### Reading From the Command Line

```python
parser = to_argparse(FrequencySweep)
# argparse will grab the command line arguments
# Ex command line: '--freqs 150 100 2 --power -40 --pulse_samples 200 --is_test'
args = parser.parse_args()
sweep_params = from_parsed_args(FrequencySweep, params_namespace=args)[0]

# even_simpler
sweep_params = create_parser_and_parse_args(FrequencySweep)
```

### Setting and getting parameters
```python
print(sweep_params.freqs)  # prints a numpy array of size (20,) array([0.0e+00, 1.0e+08, 2.0e+08, …
sweep_params.freqs = [5, 10, 5]
print(sweep_params.freqs)  # prints array([0.e+00, 2.e+09, 4.e+09, 6.e+09, 8.e+09])
print(sweep_params.is_test)  # prints True
```

### JSON and YAML serialization

```python
import json
import yaml  # pyyaml

d = to_json_serializable_dict(sweep_params)
s = json.dumps(d)
sweep_params = from_json_serializable_dict(json.loads(s))


to_yaml_file(sweep_params, 'test_params.yaml')
sweep_params = from_yaml_file('test_params.yaml')
```

### Nested Params
```python
class MultipleFreqSweeps(Params):
    sweep_1 = FrequencySweep(freqs=[0, 1, 100])
    sweep_2 = FrequencySweep(freqs=[1, 2, 100])
```
