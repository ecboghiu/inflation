Monomial classes and functions
==============================
Abstract monomial classes
-------------------------
.. autoclass:: causalinflation.quantum.monomial_classes.InternalAtomicMonomial
   :members: compute_marginal, dagger, is_physical, is_hermitian, name, symbol

.. autoclass:: causalinflation.quantum.monomial_classes.CompoundMonomial
   :members: attach_idx, compute_marginal, evaluate, is_physical, n_operators, name, symbol

InflationSDP internal monomials
-------------------------------
.. autofunction:: causalinflation.InflationSDP.Monomial
