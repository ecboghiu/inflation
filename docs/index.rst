.. inflation documentation master file, created by
   sphinx-quickstart on Tue Jun  7 17:11:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============
CausalInflation is a Python package that implements inflation algorithms for causal inference. In causal inference, the main task is to determine which causal relationships can exist between different observed random variables. Inflation algorithms are a class of techniques designed to solve the causal compatibility problem, that is, test compatiblity between some observed data and a given causal relationship.

The first version of this package implements the inflation technique for quantum causal compatibility. For details, see `Wolfe, Elie, et al. "Quantum inflation: A general approach to quantum causal compatibility." `Physical Review X 11.2 (2021): 021043. <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.021043>`_). The inflation technique for classical causal compatibility will be implemented in a future update. 

Examples of use of this package include:

- Feasibility problems and extraction of certificates.
- Optimization of Bell operators. 
- Optimisation over classical distributions. 
- Standard NPA.
- Scenarios with partial information. 

In the `Tutorial <https://ecboghiu.github.io/inflation/_build/html/tutorial.html>`_ and `Examples <https://ecboghiu.github.io/inflation/_build/html/examples.html>`_ all the above are explained in more detail.

Copyright and License
=====================

CasualInflation is free open-source software released under the `Creative Commons License <https://github.com/ecboghiu/inflation/blob/main/LICENSE>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   download
   tutorial
   examples
   advanced
   contribute
   modules
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
