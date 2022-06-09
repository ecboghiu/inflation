import os
import sys
# Locate the script in UnifiedInflation/examples and add to path one folder before, that is, UnifiedInflation/
# in order to be able to import quantuminflation
# ! Note: I found online that "__file__" sometimes can be problematic, So I'm using the solution provided in
# https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python?lq=1
from inspect import getsourcefile
from os.path import abspath
cws = abspath(getsourcefile(lambda: 0))
# Remove the script filename to get the directory of the script
cws = os.sep.join(cws.split(os.sep)[:-1])
# Go one folder above UnifiedInflation/examples -> UnifiedInflation/
cws = cws + os.sep + os.pardir
sys.path.append(cws)

from causalinflation.useful_distributions import (P_GHZ_array, P_Mermin_array,
                                                   P_W_array, P_CHSH_array)
from causalinflation.InflationSDP import InflationSDP
from causalinflation.InflationProblem import InflationProblem
import numpy as np

###############################################################################


InfProb = InflationProblem(dag={"h2": ["v1", "v2"],
                                "h3": ["v1", "v3"],
                                "h1": ["v2", "v3"]},
                           outcomes_per_party=[2, 2, 2],
                           settings_per_party=[1, 1, 1],
                           inflation_level_per_source=[2, 2, 2],
                           names=['A', 'B', 'C'])

InfSDP = InflationSDP(InfProb, commuting=False, verbose=0)
InfSDP.generate_relaxation(column_specification='npa2')

tol_vis = 1e-4
v0, v1 = 0, 1
vm = (v0 + v1)/2

InfSDP.verbose = 0

iteration = 0
certificate = 0
while abs(v1 - v0) >= tol_vis and iteration < 20:
    p = P_W_array(visibility=vm)
    InfSDP.set_distribution(p, use_lpi_constraints=True)
    InfSDP.solve()

    print("max(min eigval):", "{:10.4g}".format(
        InfSDP.primal_objective), "\tvisibility =", iteration, "{:.4g}".format(vm))
    iteration += 1
    if InfSDP.primal_objective >= 0:
        v0 = vm
        vm = (v0 + v1)/2
    elif InfSDP.primal_objective < 0:
        v1 = vm
        vm = (v0 + v1)/2
        certificate = InfSDP.dual_certificate_as_symbols_probs
    if abs(InfSDP.primal_objective) <= 1e-7:
        break

print("Final values and last valid certificate:")
print("max(min eigval):", "{:10.4g}".format(
    InfSDP.primal_objective), "\tvm =", iteration, "{:10.4g}".format(vm))
print(certificate, "≥", 0)
