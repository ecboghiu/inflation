"""
This module encodes and handles certificates of infeasibility.

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""

from warnings import warn
from typing import Union

import numpy as np
import sympy as sp

from .utils import clean_coefficients

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
        self.problem = convexOptimizationProblem
        try:
            self.certificate = self.problem.solution_object["dual_certificate"]
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call \"InflationSDP.solve()\" first.")

    def as_dict(self,
                clean: bool = True,
                chop_tol: float = 1e-10,
                round_decimals: int = 3) -> dict:
        """Give certificate as dictionary with monomials as keys and
        their coefficients in the certificate as the values. The certificate
        of incompatibility is ``cert < 0``.

        If the certificate is evaluated on a point giving a negative value, this
        guarantees that the compatibility test for the same point is infeasible
        provided the set of constraints of the program does not change. Warning:
        when using ``use_lpi_constraints=True`` the set of constraints depends
        on the specified distribution, thus the certificate is not guaranteed to
        apply.

        Parameters
        ----------
        clean : bool, optional
            If ``True``, eliminate all coefficients that are smaller than
            ``chop_tol``, normalise and round to the number of decimals
            specified by ``round_decimals``. By default ``True``.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. By default ``1e-10``.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number of
            decimals specified. By default ``3``.

        Returns
        -------
        dict
            The expression of the certificate in terms of probabilities and
            marginals. The certificate of incompatibility is ``cert < 0``.
        """
        if len(self.problem.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions.")
        if np.allclose(list(self.certificate.values()), 0.):
            return {}
        if clean:
            cert = clean_coefficients(self.certificate, chop_tol, round_decimals)
        else:
            cert = self.certificate
        return {self.problem.monomial_from_name[k]: v for k, v in cert.items()
                if not self.problem.monomial_from_name[k].is_zero}

    def as_string(self,
                  clean: bool = True,
                  chop_tol: float = 1e-10,
                  round_decimals: int = 3) -> str:
        """Give the certificate as a string of a sum of probabilities. The
        expression is in the form such that its satisfaction implies
        incompatibility.

        If the certificate is evaluated on a point giving a negative value, this
        guarantees that the compatibility test for the same point is infeasible
        provided the set of constraints of the program does not change. Warning:
        when using ``use_lpi_constraints=True`` the set of constraints depends
        on the specified distribution, thus the certificate is not guaranteed to
        apply.

        Parameters
        ----------
        clean : bool, optional
            If ``True``, eliminate all coefficients that are smaller than
            ``chop_tol``, normalise and round to the number of decimals
            specified by ``round_decimals``. By default, ``True``.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. By default, ``1e-10``.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number of
            decimals specified. By default, ``3``.

        Returns
        -------
        str
            The certificate in terms of probabilities and marginals. The
            certificate of incompatibility is ``cert < 0``.
        """
        cert_dict = self.as_dict(
            clean=clean,
            chop_tol=chop_tol,
            round_decimals=round_decimals)
        as_dict = self.problem._sanitise_dict(cert_dict)
        # Watch out for when "1" is note the same as "constant_term"
        constant_value = as_dict.pop(self.problem.Constant_Term,
                                     as_dict.pop(self.problem.One, 0.)
                                     )
        if constant_value:
            polynomial_as_str = str(constant_value)
        else:
            polynomial_as_str = ""
        for mon, coeff in as_dict.items():
            if mon.is_zero or np.isclose(np.abs(coeff), 0):
                continue
            else:
                polynomial_as_str += "+" if coeff >= 0 else "-"
                if np.isclose(abs(coeff), 1):
                    polynomial_as_str += mon.name
                else:
                    polynomial_as_str += "{0}*{1}".format(abs(coeff), mon.name)

        if polynomial_as_str[0] = "+":
            polynomial_as_str = polynomial_as_str[1:]
        return polynomial_as_str + " < 0"

    def as_probs(self,
                 clean: bool = True,
                 chop_tol: float = 1e-10,
                 round_decimals: int = 3) -> sp.core.add.Add:
        """Give certificate as symbolic sum of probabilities. The certificate
        of incompatibility is ``cert < 0``.

        If the certificate is evaluated on a point giving a negative value, this
        guarantees that the compatibility test for the same point is infeasible
        provided the set of constraints of the program does not change. Warning:
        when using ``use_lpi_constraints=True`` the set of constraints depends
        on the specified distribution, thus the certificate is not guaranteed to
        apply.

        Parameters
        ----------
        clean : bool, optional
            If ``True``, eliminate all coefficients that are smaller than
            ``chop_tol``, normalise and round to the number of decimals
            specified by ``round_decimals``. By default ``True``.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. By default ``1e-10``.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number of
            decimals specified. By default ``3``.

        Returns
        -------
        sympy.core.add.Add
            The expression of the certificate in terms of probabilities and
            marginals. The certificate of incompatibility is ``cert < 0``.
        """
        cert_dict = self.as_dict(
            clean=clean,
            chop_tol=chop_tol,
            round_decimals=round_decimals)
        polynomial = sp.S.Zero
        for mon, coeff in self.problem._sanitise_dict(cert_dict).items():
            polynomial += coeff * mon.symbol
        return polynomial

    def evaluate(self, prob_array: np.ndarray) -> float:
        """Evaluate the certificate of infeasibility in a target probability
        distribution. If the evaluation is a negative value, the distribution is
        not compatible with the causal structure. Warning: when using
        ``use_lpi_constraints=True`` the set of constraints depends on the
        specified distribution, thus the certificate is not guaranteed to apply.

        Parameters
        ----------
        prob_array : numpy.ndarray
            Multidimensional array encoding the distribution, which is
            called as ``prob_array[a,b,c,...,x,y,z,...]`` where
            :math:`a,b,c,\\dots` are outputs and :math:`x,y,z,\\dots` are
            inputs. Note: even if the inputs have cardinality 1 they must
            be specified, and the corresponding axis dimensions are 1.
            The parties' outcomes and measurements must appear in the
            same order as specified by the ``order`` parameter in the
            ``InflationProblem`` used to instantiate ``InflationLP``.

        Returns
        -------
        float
            The evaluation of the certificate of infeasibility in prob_array.
        """
        if self.problem.use_lpi_constraints:
            warn("You have used LPI constraints to obtain the certificate. " +
                 "Be aware that, because of that, the certificate may not be " +
                 "valid for other distributions.")
        return self.problem.evaluate_polynomial(self.as_dict(), prob_array)

    def desymmetrize(self) -> dict:
        """If the scenario contains symmetries other than the inflation
        symmetries, this function writes a certificate of infeasibility valid
        for non-symmetric distributions too.

        Returns
        -------
        dict
            The expression of the un-symmetrized certificate in terms of
            probabilities and marginals. The certificate of incompatibility is
            ``cert < 0``.
        """
        desymmetrized = {}
        norm = len(self.problem.InflationProblem.symmetries)
        lexmon_names = self.problem.InflationProblem._lexrepr_to_copy_index_free_names
        # TODO: Make that the lexorder in InflationLP and InflationSDP is the
        # same, so we do not need this offset
        offset = int(type(self.problem) is InflationSDP)
        for symm in self.problem.InflationProblem.symmetries:
            for mon, coeff in self.certificate.items():
                mon = self.problem.monomial_from_name[mon]
                if not mon.is_zero:
                    desymm_mon = lexmon_names[symm[mon.as_lexmon-offset]]
                    desymm_mon = sorted(desymm_mon, key=lambda x: x[0])
                    if not mon.is_one:
                        desymm_name = "P[" + " ".join(desymm_mon) + "]"
                    else:
                        desymm_name = "1"
                    if desymm_name not in desymmetrized:
                        desymmetrized[desymm_name] = coeff / norm
                    else:
                        desymmetrized[desymm_name] += coeff / norm
        return desymmetrized
