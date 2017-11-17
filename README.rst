ConjugateGradients
==================

Implementation of Conjugate Gradient method for solving systems of linear equation using Python, C and Nvidia CUDA.
Currently only Python implementation is available - it includes Conjugate Gradient Method and Preconditioned Conjugate Gradient with Jacobi
pre-conditioner (hopefully others will be added as well).

Road-map:

::

 - [X] Python implementation
        - [X] Create test matrices generator
        - [X] Implement pure CG
        - [ ] Implement PCG
                - [X] Jacobi preconditioner
                - [ ] SSOR preconditioner
                - [ ] Incomplete Cholesky factorization preconditioner

 - [ ] C implementation
        - [ ] Implement pure CG
                - [X] Implementation using BLAS library (Intel MKL*)
                - [ ] Custom BLAS implementation using OpenMP
        - [ ] Implement PCG
                - [ ] Jacobi preconditioner
                        - [ ] Implementation using BLAS library (Intel MKL*)
                        - [ ] Custom BLAS implementation using OpenMP

 - [ ] CUDA implementation
        - [ ] Implement pure CG
                - [ ] Reference implementation using CUBLAS
                - [ ] Custom kernels implementation
        - [ ] Implement PCG
                - [ ] Jacobi preconditioner
                        - [ ] Reference implementation using CUBLAS
                        - [ ] Custom kernels implementation

 * MKL can be obtained free of charge: https://software.intel.com/en-us/mkl


Getting Started
---------------

::

    $ git clone https://github.com/stovorov/ConjugateGradients
    $ cd ConjugateGradients


Python implementation
---------------------

Prepare environment
~~~~~~~~~~~~~~~~~~~

::

    $ source run_me.sh (sets PYTHONPATH)
    $ cd scripts
    $ make venv
    $ source venv/bin/activate
    $ make test


Usage
~~~~~

.. code:: python

    from random import uniform
    from scripts.ConjugateGradients.test_matrices import TestMatrices
    from scripts.ConjugateGradients.utils import get_solver

    import numpy as np

    matrix_size = 100
    # patterns are: quadratic, rectangular, arrow, noise, curve
    # pattern='qrana' means that testing matrix will be composition of all mentioned patterns
    a_matrix = TestMatrices.get_random_test_matrix(matrix_size)
    x_vec = np.vstack([1 for x in range(matrix_size)])
    b_vec = np.vstack([uniform(0, 1) for x in range(matrix_size)])
    CGSolver = get_solver('CG')             # pylint: disable=invalid-name; get_solver returns Class
    PCGJacobiSolver = get_solver('PCG')     # pylint: disable=invalid-name; get_solver returns Class
    cg_solver = CGSolver(a_matrix, b_vec, x_vec)
    cg_solver.solve()
    cg_solver.show_convergence_profile()

    pcg_solver = PCGJacobiSolver(a_matrix, b_vec, x_vec)
    pcg_solver.solve()

    CGSolver.compare_convergence_profiles(cg_solver, pcg_solver)


You can view convergence profile using solver's ``show_convergence_profile`` method:

    .. image:: doc/cg_conv_visual.png
        :height: 200 px
        :width: 200 px
        :scale: 50 %

You can compare convergence profiles of difference solvers using ``compare_convergence_profiles`` method:

    .. image:: doc/comparison.png
        :height: 200 px
        :width: 200 px
        :scale: 50 %

Different testing matrices can be generated using ``TestMatrix`` class, for more information please refer methods descriptions.
Matrices can be viewed using ``view_matrix`` function, which can be found in ``utils.py``. Below matrices are symmetric
and positively defined.

    .. image:: doc/arn_matrix.png
        :height: 200 px
        :width: 200 px
        :scale: 50 %

    .. image:: doc/crn_matrix.png
        :height: 200 px
        :width: 200 px
        :scale: 50 %

    .. image:: doc/rnqa_matrix.png
        :height: 200 px
        :width: 200 px
        :scale: 50 %

Examples can be found in ``scripts/ConjugateGradients/demo.py``
Required Python 3.5+


C implementation - TBA
----------------------

MKL implementation
~~~~~~~~~~~~~~~~~~

Required - Intel MKL library for BLAS operations. Implementation was tested on version 2017 though older should work as well.
By default MKL will be installed in directory ``/opt/intel/mkl/``.

To compile:

::

    $ source mkl_setup.sh
    $ make

``mkl_setup.sh`` sources MKL env configuration.

::

    If you have installed MKL in a different directory, then you will have to adjust mkl_setup.sh
    script to point to proper paths.
    Also please note that in makefile there is MKLROOT variable which points to MKL installation directory.

By default MKL will be compiled as a static library.
Since there are many dependencies it is good to set ``CFLAGS`` and ``LDFLAGS`` accordingly to MKL link line advisor:

https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor

Use ``make clean`` command to delete compiled build.

CUDA implementation - TBA
-------------------------

Conjugate Gradients description
-------------------------------

A bit about Conjugate Gradients and when it actually works (collection of information found over internet):

CG will work when is applied on symmetrical and positively defined matrix.

``CG is equivalent to applying the Lanczos algorithm on the given matrix with the starting vector given by the (normalized)
residual of the initial approximation.``
source: https://math.stackexchange.com/questions/882713/application-of-conjugate-gradient-method-to-non-symmetric-matrices

Resources:
~~~~~~~~~~

General overview and derivation is described on Wiki:
https://en.wikipedia.org/wiki/Derivation_of_the_conjugate_gradient_method

Though this description has a lot of shortcuts and will probably leave you with a more questions then before reading it...

A good description can be found in ``Painless Conjugate Gradient``:

https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
A bit complex work but worth reading (but requires a lot of focus...at least from me...).

A lot about ``preconditioning`` could be found here:
http://netlib.org/linalg/html_templates/node51.html
haven't read everything but may explain a lot (still, will probably leave you with a lot of questions...).
