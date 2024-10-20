Monomial classes and functions
==============================
InflationLP monomial classes
----------------------------
.. autoclass:: inflation.lp.monomial_classes.InternalAtomicMonomial
   :members: _raw_name, _name, _signature, compute_marginal

.. autoclass:: inflation.lp.monomial_classes.CompoundMoment
   :members: attach_idx, compute_marginal, evaluate

InflationSDP monomial classes
-----------------------------
.. autoclass:: inflation.sdp.monomial_classes.InternalAtomicMonomialSDP
   :members: is_all_commuting, conjugate_lexmon, is_hermitian, dagger, 

.. autoclass:: inflation.sdp.monomial_classes.CompoundMomentSDP
   :members: is_all_commuting, is_physical, is_hermitian

