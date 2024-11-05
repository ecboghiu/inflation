from inflation import InflationProblem
import numpy as np
triangle = InflationProblem(dag={"AB": ["A", "B"],
                                 "BC": ["B", "C"],
                                 "AC": ["A", "C"]},
                            outcomes_per_party=(2,2,2),
                            settings_per_party=(1,1,1),
                            inflation_level_per_source=(2,2,2),
                            order=["A","B","C"])
print(triangle._lexorder)
permuting_copy_indices = triangle._lexorder[:,[0,2,3,1,4,5]]
permuting_copy_indices[permuting_copy_indices[:,2]==0,0] = 1
permuting_copy_indices[permuting_copy_indices[:,3]==0,0] = 2
permuting_copy_indices[permuting_copy_indices[:,1]==0,0] = 3
print(np.stack((triangle._lexorder,permuting_copy_indices),axis=1))
new_lexorder_permutation = np.array([triangle._lexorder_lookup[op.tobytes()] for op in permuting_copy_indices])
print(triangle.lexorder_symmetries)
all_symmetries_new = np.vstack((triangle.lexorder_symmetries,
                                triangle.lexorder_symmetries[:,new_lexorder_permutation],
                                triangle.lexorder_symmetries[:,new_lexorder_permutation[new_lexorder_permutation]]))
triangle.lexorder_symmetries = all_symmetries_new
print(triangle.lexorder_symmetries)
# print(triangle._lexrepr_to_names)