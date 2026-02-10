Lanczos Method
==============

The Lanczos algorithm is an iterative method for finding a subset of 
eigenvalues and eigenvectors of large sparse matrices. This implementation is
a *block* variant of the standard algorithm, i.e., instead of iterating with single
vectors it iterates with blocks of vectors (the standard algorithm can of course be
recovered by using blockvectors with block size one).

Additionally, it implements the so-called thick restart variant: when the algorithm
has extended the Lanczos basis to the maximum allowed size (as given by the user)
it restarts the method using the information contained in the current basis.

.. doxygenclass:: trl::BlockLanczos
   :project: trl
   :members:
