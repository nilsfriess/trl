#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <span>

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
    std::generate_n(V.data, V.rows() * V.cols(), [&]() { return dist(rng); });
  }

  typename EVP::Scalar norm(typename EVP::BlockMultivector::BlockView V)
  {
    typename EVP::Scalar norm = 0;
    for (std::size_t i = 0; i < V.rows() * V.cols(); ++i) norm += V.data[i] * V.data[i];
    return std::sqrt(norm);
  }

  std::vector<typename EVP::Scalar> to_host_data(typename EVP::BlockMultivector::BlockMatrix::BlockView B)
  {
    sync();
    std::vector<typename EVP::Scalar> host_data(B.rows * B.cols);
    std::copy_n(B.data, host_data.size(), host_data.data());
    return host_data;
  }

  std::vector<typename EVP::Scalar> to_host_data(std::span<typename EVP::Scalar, std::dynamic_extent> data)
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
