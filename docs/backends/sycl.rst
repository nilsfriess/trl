SYCL Backend
============

This backend enables GPU-accelerated computation through SYCL. It is only tested with
`AdaptiveCpp <https://github.com/AdaptiveCpp/AdaptiveCpp>`_ but should work with other
SYCL implementations as well

.. warning::
   The SYCL backend is currently experimental and under active development.

API Reference
-------------

Like every backend, the SYCL backend implements classes representing block multivectors and block matrices and views of individual blocks for these types.

.. doxygenclass:: trl::sycl::BlockMultivector
   :project: trl
   :members:

.. doxygenclass:: trl::sycl::BlockView
   :project: trl
   :members:

.. doxygenclass:: trl::sycl::BlockMatrix
   :project: trl
   :members:

.. doxygenclass:: trl::sycl::MatrixBlockView
   :project: trl
   :members:      
