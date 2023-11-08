from inflation import InflationProblem

bilocal_scenario_with_inflation = InflationProblem({
    "lambda": ["a", "b"],
    "mu": ["b", "c"]},
    outcomes_per_party=[2, 2, 2],
    settings_per_party=[2, 1, 2],
    inflation_level_per_source=[2, 2],
    order=['a', 'b', 'c'],
    verbose=1)

print(bilocal_scenario_with_inflation._possible_party_relabelling_symmetries())
print(bilocal_scenario_with_inflation._lexorder)
print(bilocal_scenario_with_inflation._possible_setting_specific_outcome_relabelling_symmetries())

#TODO
_possible_party_specific_setting_relabelling_symmetries()