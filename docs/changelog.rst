*********
Changelog
*********

1.1.0 - 2023-09-23
******************

* Added support for linear programming relaxations of causal scenarios, supporting hybrid networks with either classical sources of correlations or general no-signaling sources of correlations, and possibilistic-type feasibility problems.
* Linear programming relaxations use the low-level Python MOSEK Optimizer API to reduce overhead, the Collins-Gisin formulation of probablities to reduce the number of variables, and sparse matrices to represent all constraints.

1.0.0 - 2022-11-28
******************

* Initial release.