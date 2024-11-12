*********
Changelog
*********

2.0.1 - 2024-10-21
******************

* Support for `NumPy 2 <https://numpy.org/devdocs/release/2.0.0-notes.html>`_.

* Small bugfixes.

* Memory and runtime improvements.

2.0.0 - 2024-06-01
******************

* Added support for linear programming relaxations of causal scenarios, via ``InflationLP``. This allows to run inflation hierarchies bounding the sets of classical and no-signaling correlations, per `J. Causal Inference 7(2), 2019 <https://doi.org/10.1515/jci-2017-0020>`_ (`arXiv:1609.00672 <https://arxiv.org/abs/1609.00672>`_)

* Added support for hybrid scenarios with sources of different nature, via the ``classical_sources`` argument to ``InflationProblem``. Currently supported: classical-no-signaling (via ``InflationLP``) and classical-quantum (via ``InflationSDP``).

* Added support for possibilistic-type feasibility problems (via ``supports_problem`` in ``InflationLP`` and ``InflationSDP``).

* Added initial support for structures with multiple layers of latent variables.

* Improved support for structures with visible-to-visible connections, using the notation of do-conditionals.

* Improved handling of certificates. This makes them easier to manipulate and evaluate.

* Revamped the description of monomials. This makes the codes faster and consume less memory.

1.0.0 - 2022-11-28
******************

* Initial release.
