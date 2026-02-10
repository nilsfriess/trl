trl - Thick Restart (Block) Lanczos
========================================

A C++20 implementation of a Block Lanczos eigensolver with thick restarting scheme 
for computing eigenvalues and eigenvectors of large sparse matrices.

The algorithm itself is implemented using an abstract concept of an eigenvalue problem
which itself must define an abstract concept of a block multivector (and related types).
This makes it possible to adapt the method to different hardware (e.g. CPUs or GPUs) without
changing the algorithm itself.

Currently two implementations (here called "backends") are available that can be plugged
into the algorithm:

- An OpenMP backend: 
  This is the default and is always built. It requires and OpenMP capable compiler.
  
- A SYCL backend: 
  This backend is experimental and work in progress. It is currently only tested with the
  `AdaptiveCpp <https://github.com/AdaptiveCpp/AdaptiveCpp>`_ SYCL compiler but other
  SYCL implementations should work as well.

To facilitate the implementation of new backends, C++20 concepts for the eigenproblem and
the necessary types are given.

.. toctree::
   :maxdepth: 1
   :caption: Backends:

   backends/openmp
   backends/sycl

.. toctree::
   :maxdepth: 1
   :caption: C++ 20 Concepts

   concepts/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Algorithms:

   algorithms/lanczos

..
   Design
   ------

   The library is designed with a flexible backend architecture that allows different 
   multivector implementations to be easily integrated. This enables the use of 
   specialized hardware implementations, such as GPU-accelerated backends, without 
   modifying the core algorithm.

   Backends
   --------

   Currently provides:

   * OpenMP backend for CPUs
   * SYCL backend using AdaptiveCpp (experimental)

   Building
   --------

   Requirements:

   * CMake
   * C++20 compatible compiler
   * Eigen
   * Optionally AdaptiveCpp for the SYCL backend

   Build the OpenMP backend:

   .. code-block:: bash

      mkdir build && cd build
      cmake ..
      make

   Run tests:

   .. code-block:: bash

      ctest

   Indices and tables
   ==================

..
   * :ref:`search`
