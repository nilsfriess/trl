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

Reorthogonalization
-------------------

Full reorthogonalization is performed at each Lanczos step using a pluggable
strategy, passed as the second template parameter of :cpp:class:`trl::BlockLanczos`
(default: :cpp:struct:`trl::ModifiedGS`). Custom strategies must satisfy the
:cpp:concept:`trl::ReorthogonalizationStrategy` concept (see :doc:`/concepts/index`).

.. doxygenstruct:: trl::ModifiedGS
   :project: trl

API Reference
-------------

.. doxygenclass:: trl::BlockLanczos
   :project: trl
   :members:
