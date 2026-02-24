OpenMP Backend
==============

The OpenMP backend provides CPU-based parallel computation using OpenMP.
It is the default backend and always available.

API Reference
-------------

Like every backend, the OpenMP backend implements classes representing block multivectors and block matrices and views of individual blocks for these types.

.. doxygenclass:: trl::openmp::BlockMultivector
   :project: trl
   :members:

.. doxygenclass:: trl::openmp::BlockView
   :project: trl
   :members:

.. doxygenclass:: trl::openmp::BlockMatrix
   :project: trl
   :members:

.. doxygenclass:: trl::openmp::BlockMatrixBlockView
   :project: trl
   :members:      

