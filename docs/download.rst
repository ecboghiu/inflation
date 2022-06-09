*************************
Download and Installation
*************************

Dependencies
============
The implementation requires `SymPy <http://sympy.org/>`_ and `Numpy <http://www.numpy.org/>`_. The code is only compatible with Python 3.

We use the `MOSEK <http://www.mosek.com/>` solver for solving the semidefinite program. There exist free academic licenses for this solver.
The MOSEK Python package is required.

Installation
============
Follow the standard procedure for installing Python modules:

::

    $ pip install ncpol2sdpa

If you use the development version, install it from the source code:

::

    $ git clone https://github.com/peterwittek/ncpol2sdpa.git
    $ cd ncpol2sdpa
    $ python setup.py install