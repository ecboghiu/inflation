.. inflation documentation master file, created by
   sphinx-quickstart on Tue Jun  7 17:11:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

************
Introduction
************
Inflation is a Python package that implements inflation algorithms for causal inference. In causal inference, the main task is to determine which causal relationships can exist between different observed random variables. Inflation algorithms are a class of techniques designed to solve the causal compatibility problem, that is, test compatibility between some observed data and a given causal relationship.

Version 1.0.0 of this package implements the inflation technique for quantum causal compatibility. For details, see `Wolfe et al. "Quantum inflation: A general approach to quantum causal compatibility." Physical Review X 11 (2), 021043 (2021) <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.021043>`_. The inflation technique for classical causal compatibility will be implemented in a future update.

Examples of use of this package include:

- Feasibility problems and extraction of certificates.
- Optimization of Bell operators.
- Optimization over classical distributions.
- Standard `Navascues-Pironio-Acin hierarchy <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.010401>`_.
- Scenarios with partial information.

In the `Tutorial <https://ecboghiu.github.io/inflation/_build/html/tutorial.html>`_ and `Examples <https://ecboghiu.github.io/inflation/_build/html/examples.html>`_ all the above are explained in more detail.

Copyright and License
=====================

CasualInflation is free open-source software released under the `GNU General Public License <https://github.com/ecboghiu/inflation/blob/main/LICENSE>`_.

How to cite
===========
If you use Inflation in your work, please cite `Inflation's paper <https://www.arxiv.org/abs/2211.04483>`_:

  Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens, "Inflation: a Python package for classical and quantum causal compatibility", arXiv:2209.xxxxx

.. code-block:: html

    @misc{2211.04483,
      doi = {10.48550/arxiv.2211.04483},
      url = {https://arxiv.org/abs/2211.04483},
      author = {Boghiu, Emanuel-Cristian and Wolfe, Elie and Pozas-Kerstjens, Alejandro},
      title = {{Inflation}: a {Python} package for classical and quantum causal compatibility},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }

.. toctree::
