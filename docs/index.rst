.. inflation documentation master file, created by
   sphinx-quickstart on Tue Jun  7 17:11:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============
Causalinflation is a Python package that provides implementations of various inflation algorithms for causal inference. In causal inference, the main task is to determine which types of causal relationships can exist between different random variables. Inflation algorithms allow one to exclude, based on data, certain causal relationships. 

The first version of this package implements the inflation technique for Quantum Causal Compatibility (see `Wolfe, Elie, et al. "Quantum inflation: A general approach to quantum causal compatibility." Physical Review X 11.2 (2021): 021043. <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.021043>`_). The inflation techniques for classical and post-quantum causal compatibility will be implemented in future updates. 

Examples of applications of this package:

- Standard NPA .. expand
-

The implementation has an intuitive syntax and the user does not require advanced knowledge of the inner workings of the package.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   download
   tutorial
   examples
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
