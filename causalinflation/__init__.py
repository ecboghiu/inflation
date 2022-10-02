"""CausalInflation
==================
Provides
 1. A tool for describing causal scenarios and reducing them to network form
    (see the definitions of network and non-network causal scenarios in
    arXiv:1707.06476 and arXiv:1909.10519).
 2. A tool for setting up and solving feasibility and optimization problems over
    probability distributions compatible with quantum causal scenarios.
"""

__version__ = "0.1"

from .InflationProblem import InflationProblem
from .quantum.InflationSDP import InflationSDP
from ._about import about
from ._version import __version__

from typing import Dict, List, Tuple, Union, Any, NewType, Iterable

__all__ = ["InflationProblem",
           "InflationSDP"]
