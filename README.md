# CausalInflation
CausalInflation is a Python package that implements inflation algorithms for causal inference. In causal inference, the main task is to determine which causal relationships can exist between different observed random variables. Inflation algorithms are a class of techniques designed to solve the causal compatibility problem, that is, test compatiblity between some observed data and a given causal relationship.

The first version of this package implements the inflation technique for quantum causal compatibility. For details, see [Physical Review X 11.2 (2021): 021043](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.021043). The inflation technique for classical causal compatibility will be implemented in a future update. 

Examples of use of this package include:

- Feasibility problems and extraction of certificates.
- Optimization of Bell operators. 
- Optimisation over classical distributions. 
- Standard NPA.
- Scenarios with partial information. 

See the documentation for more details.

## Documentation

* [Latest version](https://ecboghiu.github.io/inflation/).

## Installation

To install the package, run the following command:

```
pip install causalinflation
```

You can also install the latest developed version with:

`pip install git+https://github.com/ecboghiu/inflation.git@main`

or clone the repository and run `python setup.py install` in the downloaded folder.

## Getting started

Below is a simple complete ready-to-run example that shows that the W distribution is incompatible with the triangle scenario with quantum sources:

```
from causalinflation import InflationProblem, InflationSDP
import numpy as np
import itertools

P_W = np.zeros((2, 2, 2, 1, 1, 1))
x, y, z = 0, 0, 0
for a, b, c in itertools.product([0, 1], repeat=3):
    if a + b + c == 1:
        P_W[a, b, c, x, y, z] = 1 / 3

scenario = InflationProblem(dag={"rho_AB": ["A", "B"],
                                 "rho_BC": ["B", "C"],
                                 "rho_AC": ["A", "C"]}, 
                             outcomes_per_party=[2, 2, 2],
                             settings_per_party=[1, 1, 1],
                             inflation_level_per_source=[2, 2, 2])

sdprelax = InflationSDP(scenario)
sdprelax.generate_relaxation('npa2')
sdprelax.set_distribution(P_W)
sdprelax.solve()

print(sdprelax.status)
```

For more information about the theory and other features, please visit the [documentation](https://ecboghiu.github.io/inflation/), and more specifically the [Tutorial](https://ecboghiu.github.io/inflation/_build/html/tutorial.html) and [Examples](https://ecboghiu.github.io/inflation/_build/html/examples.html) pages.

## How to contribute

If you want to contribute, visit the [documentation](https://ecboghiu.github.io/inflation/) to learn how the package works, and then submit an issue or create a pull request on the GitHub repository.

## License

CasualInflation is free open-source software released under the [Creative Commons License](https://github.com/ecboghiu/inflation/blob/main/LICENSE>).
