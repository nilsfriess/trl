# Block Lanczos Eigensolver

A C++20 implementation of a Block Lanczos eigensolver with thick restarting scheme for computing eigenvalues and eigenvectors of large sparse matrices.

## Design

The library is designed with a flexible backend architecture that allows different multivector implementations to be easily integrated. This enables the use of specialized hardware implementations, such as GPU-accelerated backends, without modifying the core algorithm.

## Backends

Currently provides an OpenMP backend for CPUs and a SYCL backend using [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (other SYCL implementations probably work as well but are not tested). The SYCL backend is experimental.

## Building

Requires:
- CMake
- C++20 compatible compiler
- [Eigen](https://libeigen.gitlab.io/)
- Optionally AdaptiveCpp for the SYCL backend

If only the OpenMP backend should be built:
```bash
mkdir build && cd build
cmake ..
make
```

Run tests:
```bash
ctest
```
