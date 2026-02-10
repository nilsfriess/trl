#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <span>
#include <vector>

#include <trl/concepts.hh>

template <trl::Eigenproblem EVP>
class OpenMPTestHelper {
public:
  void set_random(typename EVP::BlockMultivector::BlockView V)
  {
    std::generate_n(V.data_, V.rows() * V.cols(), [&]() { return dist(rng); });
  }

  typename EVP::Scalar norm(typename EVP::BlockMultivector::BlockView V)
  {
    typename EVP::Scalar norm = 0;
    for (std::size_t i = 0; i < V.rows() * V.cols(); ++i) norm += V.data_[i] * V.data_[i];
    return std::sqrt(norm);
  }

  std::vector<typename EVP::Scalar> to_host_data(typename EVP::BlockMultivector::BlockMatrix::BlockView B)
  {
    std::vector<typename EVP::Scalar> host_data(EVP::blocksize * EVP::blocksize);
    std::copy_n(B.data_, host_data.size(), host_data.data());
    return host_data;
  }

  std::vector<typename EVP::Scalar> to_host_data(const std::vector<typename EVP::Scalar>& data) { return data; }

  std::vector<typename EVP::Scalar> to_host_data(std::span<const typename EVP::Scalar> data)
  {
    std::vector<typename EVP::Scalar> host_data(data.size());
    std::copy_n(data.data(), data.size(), host_data.data());
    return host_data;
  }

  void sync() {}

private:
  std::mt19937 rng;
  std::normal_distribution<typename EVP::Scalar> dist;
};
