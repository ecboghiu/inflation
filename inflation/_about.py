import platform


from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version
from sympy import __version__ as sympy_version

import inflation


def about() -> None:
    """Displays information about Inflation, core/optional packages, and
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
Inflation: Implementations of the Inflation Technique for Causal Inference
==============================================================================
Authored by: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens

Inflation Version:\t{inflation.__version__}

Core Dependencies
-----------------
NumPy Version:\t{numpy_version}
SciPy Version:\t{scipy_version}
SymPy Version:\t{sympy_version}
Numba Version:\t{numba_version}
Mosek Version:\t{mosek_version}

Python Version:\t{platform.python_version()}
Platform Info:\t{platform.system()} ({platform.machine()})
"""
    print(about_str)


if __name__ == "__main__":
    about()
