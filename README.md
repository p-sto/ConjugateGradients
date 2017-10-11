ConjugateGradients
==================

Implementation of Conjugate Gradient method for solving systems of linear equation using Python and Nvidia CUDA.

A bit about Conjugate Gradients and when it actually works (collection of information found over internet):

CG will work when is applied on symmetrical and positively defined matrix.

``CG is equivalent to applying the Lanczos algorithm on the given matrix with the starting vector given by the (normalized)
residual of the initial approximation.``
source: https://math.stackexchange.com/questions/882713/application-of-conjugate-gradient-method-to-non-symmetric-matrices


Resources:

General overview and derivation is described on Wiki:
https://en.wikipedia.org/wiki/Derivation_of_the_conjugate_gradient_method

Though this description has a lot of shortcuts and will probably leave you with a more questions then before reading it...

A good description can be found in ``Painless Conjugate Gradient``:

https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
A bit complex work but worth reading (but requires a lot of focus...at least from me...).

A lot about ``preconditioning`` could be found here:
http://netlib.org/linalg/html_templates/node51.html
haven't read everything but may explain a lot (still, will probably leave you with a lot of questions...).
