import platform


from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version
from sympy import __version__ as sympy_version

import causalinflation


def about() -> None:
    """Displays information about CausalInflation, core/optional packages, and
    Python version/platform information.
    """
    try:
        from numba import __version__ as numba_version
    except ImportError:
        numba_version = "Not installed"
    try:
        from mosek import Env
        major, minor, revision = Env.getversion()
        mosek_version = f"{major}.{minor}.{revision}"
    except ImportError:
        mosek_version = "Not installed"

    about_str = f"""
CausalInflation: Implementations of the Inflation Technique for Causal Inference
================================================================================
Authored by: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens

CausalInflation Version:\t{causalinflation.__version__}

Core Dependencies
-----------------
NumPy Version:\t{numpy_version}
SciPy Version:\t{scipy_version}
SymPY Version:\t{sympy_version}
Numba Version:\t{numba_version}
Mosek Version:\t{mosek_version}

Python Version:\t{platform.python_version()}
Platform Info:\t{platform.system()} ({platform.machine()})
"""
    print(about_str)

def cite():
    cite_str = """
    @article{CausalInflation: a Python library for classical and quantum causal compatibility,
  doi = {10.22331/q-2021-07-13-499},
  url = {https://doi.org/10.22331/q-2021-07-13-499},
  title = {Single-copy activation of {B}ell nonlocality via broadcasting of quantum states},
  author = {Boghiu, Emanuel-Cristian and Wolfe, Elie and Pozas-Kerstjens, Alejandro},
  journal = {{Quantum}},
  issn = {1}
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {1},
  pages = {1},
  month = jul,
  year = {1}
}
    """
    return cite_str

if __name__ == "__main__":
    about()
