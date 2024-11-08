from inflation import InflationProblem, InflationLP, InflationSDP
import numpy as np
from itertools import permutations

d=3

triangle = InflationProblem(dag={"AB": ["A", "B"],
                                 "BC": ["B", "C"],
                                 "AC": ["A", "C"]},
                            outcomes_per_party=(d,d,d),
                            settings_per_party=(1,1,1),
                            inflation_level_per_source=(2,2,2),
                            order=["A","B","C"],
                            classical_sources='all')

print(triangle.original_dag_events)
# print([event for event in triangle.original_dag_events])
# print([triangle._interpret_operator(event) for event in triangle.original_dag_events])
# print(triangle._original_event_names)

victors_distribution = np.zeros(shape=(d,d,d,1,1,1), dtype=float)
for a in range(d):
    for b in range(0,a):
        for c in range(0,b):
            val = (1-1/d)/(d*(d-1)*(d-2))
            indices_choices = permutations((a,b,c))
            for indices in indices_choices:
                victors_distribution[indices] = val
    victors_distribution[a,a,a] = 1/d*1/d

# print("Victor's probabilities:", np.unique(victors_distribution))

## BETA: We have (hopefully) automated the process of identifying all relevant
## physical symmetries given a target distribution.

(discovered_symmetries_lexorder, discovered_symmetries_original) = triangle.discover_distribution_symmetries(victors_distribution)
# for sym in discovered_symmetries_original:
#     new_sym = {str(orig): str(new) for (orig, new) in triangle._interpret_original_symmetry(sym).items() if not orig==new}
#     print(new_sym)


triangle.lexorder_symmetries = triangle._incorporate_new_symmetries(discovered_symmetries_lexorder)
print("Group order: ", len(triangle.lexorder_symmetries))


triangleLP = InflationLP(triangle, verbose=2, supports_problem=False)
triangleLP.set_distribution(victors_distribution, use_lpi_constraints=True)
triangleLP.solve()
print(triangleLP.status)

# for monomial in triangleLP.atomic_monomials:
#     if monomial.n_operators < 3:
#         print(monomial)
#
# triangleLP.set_bounds({
#     "P[A=0]": 1,
#     # "P[A=1]": 1,
#     # "P[A=2]": 1
# }, bound_type="lo")
# triangleLP.update_values({
#     "P[A=0 B=0]": 0,
#     # "P[A=1 B=1]": 0,
#     # "P[A=2 B=2]": 0
# }, use_lpi_constraints=True)
# solution_x = triangleLP.solution_object["x"]
# print("These should be zero:")
# print(solution_x["P[A=0 B=0]"])
# print(solution_x["P[A=0 B=0 C=0]"])
# print(solution_x["P[A=0 B=0 C=1]"])
# print("These should be greater than 1:")
# print(solution_x["P[A=0]"])
# print(solution_x["P[A=0 B=1]"])
# # print(solution_x["P[A=0 B=2]"])
# # print(solution_x["P[A=1 B=2]"])
# # print(solution_x["P[A=0 C=1]"])
# # print(solution_x["P[A=0 C=1]"])
# # print(solution_x["P[A=1 C=2]"])
# print(solution_x["P[A=0 B=1 C=2]"])
#
#
#
# # triangleSDP = InflationSDP(triangle, verbose=2, include_all_outcomes=False, supports_problem=True)
# # triangleSDP.generate_relaxation("local1")
# # triangleSDP.set_distribution(victors_distribution, use_lpi_constraints=True)
# # triangleSDP.solve()
# # print(triangleSDP.status)
#
# # for moment in triangleSDP.atomic_monomials:
# #     if moment.n_operators < 3:
# #         print(moment)
# #
# # triangleSDP.set_bounds({
# #     "P[A=0]": 1,
# #     "P[A=1]": 1,
# #     "P[A=2]": 1}, bound_type="lo")
# # triangleSDP.update_values({
# #     "P[A=0 B=0]": 0,
# #     "P[A=1 B=1]": 0,
# #     "P[A=2 B=2]": 0
# # }, use_lpi_constraints=True)
# # triangleSDP.solve()