"""
This module encodes and handles certificates of infeasibility.

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""

class Certificate(object):
    """
    Class for encoding relevant details concerning the causal compatibility.
    """
    def __init__(self,
                 convexOptimizationProblem: Union["InflationLP", "InflationSDP"]
                 ) -> None:
        """
        Parameters
        ----------
        convexOptimizationProblem : Union[InflationLP, InflationSDP]
            The problem from which we take the certificate of infeasibility.
        """

    def as_dict(self,
                clean: bool = True,
                chop_tol: float = 1e-10,
                round_decimals: int = 3) -> dict:

    def as_string(self,
                  clean: bool = True,
                  chop_tol: float = 1e-10,
                  round_decimals: int = 3) -> str:

    def as_probs(self,
                 clean: bool = True,
                 chop_tol: float = 1e-10,
                 round_decimals: int = 3) -> sp.core.add.Add:

    def evaluate(self, prob_array: np.ndarray) -> float:

    def desymmetrize(self) -> dict:
