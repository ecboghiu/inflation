from causalinflation.useful_distributions import P_W_array
from causalinflation import InflationSDP, InflationProblem
import numpy as np

###############################################################################


InfProb = InflationProblem(dag={"h2": ["v1", "v2"],
                                "h3": ["v1", "v3"],
                                "h1": ["v2", "v3"]},
                           outcomes_per_party=[2, 2, 2],
                           settings_per_party=[1, 1, 1],
                           inflation_level_per_source=[2, 2, 2],
                           names=['A', 'B', 'C'])

InfSDP = InflationSDP(InfProb, commuting=False, verbose=1)
cols_sym, cols_num = InfSDP.build_columns('local1') 
InfSDP.generate_relaxation(column_specification=cols_sym)

tol_vis = 1e-4
v0, v1 = 0, 1
vm = (v0 + v1)/2

InfSDP.verbose = 0

iteration = 0
certificate = 0
while abs(v1 - v0) >= tol_vis and iteration < 20:
    p = P_W_array(visibility=vm)
    InfSDP.set_distribution(p, use_lpi_constraints=True)
    InfSDP.solve(feas_as_optim=True)

    print("max(min eigval):", "{:10.4g}".format(
        InfSDP.primal_objective), "\tvisibility =", iteration, "{:.4g}".format(vm))
    iteration += 1
    if InfSDP.primal_objective >= 0:
        v0 = vm
        vm = (v0 + v1)/2
    elif InfSDP.primal_objective < 0:
        v1 = vm
        vm = (v0 + v1)/2
        certificate1 = InfSDP.dual_certificate
    if abs(InfSDP.primal_objective) <= 1e-7:
        break

print("Final values and last valid certificate:")
print("max(min eigval):", "{:10.4g}".format(
    InfSDP.primal_objective), "\tvm =", iteration, "{:10.4g}".format(vm))
print(certificate1, "≥", 0)
