[![DOI](https://zenodo.org/badge/500850617.svg)](https://zenodo.org/badge/latestdoi/500850617)

# Inflation
Inflation is a Python package that implements inflation algorithms for causal inference. In causal inference, the main task is to determine which causal relationships can exist between different observed random variables. Inflation algorithms are a class of techniques designed to solve the causal compatibility problem, that is, test compatibility between some observed data and a given causal relationship.

The first version of this package implements the inflation technique for quantum causal compatibility. For details, see [Physical Review X 11 (2), 021043 (2021)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.021043). The inflation technique for classical causal compatibility will be implemented in a future update.

Examples of use of this package include:

- Feasibility problems and extraction of certificates.
- Optimization of Bell operators.
- Optimisation over classical distributions.
- Standard [Navascués-Pironio-Acín hierarchy](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.010401).
- Scenarios with partial information.

See the documentation for more details.

## Documentation

* [Latest version](https://ecboghiu.github.io/inflation/).

## Installation

To install the package, run the following command:

```
pip install inflation
```

You can also install directly from GitHub with:

```
pip install git+https://github.com/ecboghiu/inflation.git@main
```

or download the repository on your computer and run `pip install .` in the repository folder. Install the `devel` branch for the latest features and bugfixes.

Tests are written outside the Python module, therefore they are not installed together with the package. To test the installation, clone the repository and run, in a Unix terminal,
```python -m unittest -v```
inside the repository folder.

## Example

The following example shows that the W distribution is incompatible with the triangle scenario with quantum sources by showing that a specific semidefinite programming relaxation is infeasible:

```python
from inflation import InflationProblem, InflationSDP
import numpy as np

P_W = np.zeros((2, 2, 2, 1, 1, 1))
for a, b, c, x, y, z in np.ndindex(*P_W.shape):
    if a + b + c == 1:
        P_W[a, b, c, x, y, z] = 1 / 3

triangle = InflationProblem(dag={"rho_AB": ["A", "B"],
                                 "rho_BC": ["B", "C"],
                                 "rho_AC": ["A", "C"]},
                             outcomes_per_party=(2, 2, 2),
                             settings_per_party=(1, 1, 1),
                             inflation_level_per_source=(2, 2, 2))

sdp = InflationSDP(triangle, verbose=1)
sdp.generate_relaxation('npa2')
sdp.set_distribution(P_W)
sdp.solve()

print("Problem status:", sdp.status)
print("Infeasibility certificate:", sdp.certificate_as_probs())
```

For more information about the theory and other features, please visit the [documentation](https://ecboghiu.github.io/inflation/), and more specifically the [Tutorial](https://ecboghiu.github.io/inflation/_build/html/tutorial.html) and [Examples](https://ecboghiu.github.io/inflation/_build/html/examples.html) pages.

## How to contribute

Contributions are welcome and appreciated. If you want to contribute, you can read the [contribution guidelines](https://github.com/ecboghiu/inflation/blob/main/CONTRIBUTE.md). You can also read the [documentation](https://ecboghiu.github.io/inflation/) to learn more about how the package works.

## License

Inflation is free open-source software released under [GNU GPL v. 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Citing Inflation

If you use Inflation in your work, please cite [Inflation's paper](https://www.arxiv.org/abs/2211.04483):

- Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens, "Inflation: a Python package for classical and quantum causal compatibility", arXiv:2211.04483

```
@misc{2211.04483,
  doi = {10.48550/arXiv.2211.04483},
  url = {https://arxiv.org/abs/2211.04483},
  author = {Boghiu, Emanuel-Cristian and Wolfe, Elie and Pozas-Kerstjens, Alejandro},
  title = {{Inflation}: a {Python} package for classical and quantum causal compatibility},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
