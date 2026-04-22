#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <span>
#include <vector>

#include <sycl/sycl.hpp>

#include "trl/concepts.hh"

template <trl::Eigenproblem EVP>
class SYCLTestHelper {
public:
  SYCLTestHelper(sycl::queue queue)
      : queue(queue)
  {
  }

  void set_random(typename EVP::BlockMultivector::BlockView V)
  {
    const std::size_t n = V.rows() * V.cols();
    std::vector<typename EVP::Scalar> host(n);
    std::generate_n(host.begin(), n, [&]() { return dist(rng); });
    queue.memcpy(V.data, host.data(), n * sizeof(typename EVP::Scalar)).wait();
  }

  typename EVP::Scalar norm(typename EVP::BlockMultivector::BlockView V)
  {
    const std::size_t n = V.rows() * V.cols();
    std::vector<typename EVP::Scalar> host(n);
    queue.memcpy(host.data(), V.data, n * sizeof(typename EVP::Scalar)).wait();
    typename EVP::Scalar s = 0;
    for (std::size_t i = 0; i < n; ++i) s += host[i] * host[i];
    return std::sqrt(s);
  }

  std::vector<typename EVP::Scalar> to_host_data(typename EVP::BlockMultivector::BlockMatrix::BlockView B)
  {
    sync();
    std::vector<typename EVP::Scalar> host_data(B.rows * B.cols);
    queue.memcpy(host_data.data(), B.data, host_data.size() * sizeof(typename EVP::Scalar)).wait();
    return host_data;
  }

  std::vector<typename EVP::Scalar> to_host_data(std::span<const typename EVP::Scalar, std::dynamic_extent> data)
  {
    sync();
    std::vector<typename EVP::Scalar> host_data(data.size());
    queue.memcpy(host_data.data(), data.data(), data.size_bytes()).wait();
    return host_data;
  }

  void sync() { queue.wait(); }

private:
  sycl::queue queue;
  std::mt19937 rng;
  std::normal_distribution<typename EVP::Scalar> dist;
};
