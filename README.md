# Block Lanczos Eigensolver

A C++20 implementation of a Block Lanczos eigensolver with thick restarting scheme for computing eigenvalues and eigenvectors of large sparse matrices.

## Design

The library is designed with a flexible backend architecture that allows different multivector implementations to be easily integrated. This enables the use of specialized hardware implementations, such as GPU-accelerated backends, without modifying the core algorithm.

## Current Implementation

Currently provides a SYCL backend using the oneMath library for GPU acceleration.

## Status

Work in progress.

## Building

Requires:
- CMake 3.16 or later
- C++20 compatible compiler
- Intel SYCL
- oneMath library

```bash
mkdir build && cd build
cmake ..
make
```

Run tests:
```bash
ctest
```
