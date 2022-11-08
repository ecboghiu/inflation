"""Inflation
==================
Provides
 1. A tool for describing causal scenarios and reducing them to network form
    (see the definitions of network and non-network causal scenarios in
    arXiv:1707.06476 and arXiv:1909.10519).
 2. A tool for setting up and solving feasibility and optimization problems
    over probability distributions compatible with quantum causal scenarios.
"""

from .InflationProblem import InflationProblem
from .sdp.InflationSDP import InflationSDP
from .sdp.optimization_utils import max_within_feasible
from ._about import about
from ._version import __version__

__all__ = ["InflationProblem",
           "InflationSDP",
           "max_within_feasible"]
