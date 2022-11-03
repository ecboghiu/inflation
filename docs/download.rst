*************************
Download and Installation
*************************

The package is available in the Python Package Index. The latest development version is available on GitHub.

Dependencies
============
The implementation requires `SymPy <http://sympy.org/>`_, `Numpy <http://www.numpy.org/>`_ and `Scipy <http://www.scipy.org/>`_. The code is only compatible with Python 3. It is recommended to install `Numba <http://www.numba.org/>`_ which provides just-in-time compilation of several core functions, yielding a significant speed-up.

Generated relaxations can be exported in `SDPA format <http://euler.nmt.edu/~brian/sdplib/FORMAT>`_, which can be loaded with other solvers. We also support the use of the `MOSEK <http://www.mosek.com/>`_ solver for efficiently solving the relaxation (MOSEK has `free academic licenses <https://www.mosek.com/products/academic-licenses/>`_). The MOSEK Python package needs to be installed. Future updates will include exporting to other solver interfaces, such as CVXPY, PICOS or YALMIP.

Installation
============
Follow the standard procedure for installing Python modules:

::

    $ pip install inflation

If you use the development version, you can download the source code from the `GitHub repository <https://github.com/ecboghiu/inflation>`_ and run

::

    $ python setup.py install

in the downloaded folder.
