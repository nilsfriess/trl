#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <span>
#include <vector>

#include <trl/concepts.hh>

template <trl::BackendConcept B>
class OpenMPTestHelper {
public:
  void set_random(typename B::Multivector::BlockView V)
  {
    std::generate_n(V.data_, V.rows() * V.cols(), [&]() { return dist(rng); });
  }

  typename B::Scalar norm(typename B::Multivector::BlockView V)
  {
    typename B::Scalar norm = 0;
    for (std::size_t i = 0; i < V.rows() * V.cols(); ++i) norm += V.data_[i] * V.data_[i];
    return std::sqrt(norm);
  }

  std::vector<typename B::Scalar> to_host_data(typename B::Multivector::BlockMatrix::BlockView U)
  {
    std::vector<typename B::Scalar> host_data(B::blocksize * B::blocksize);
    std::copy_n(U.data_, host_data.size(), host_data.data());
    return host_data;
  }

  std::vector<typename B::Scalar> to_host_data(const std::vector<typename B::Scalar>& data) { return data; }

  std::vector<typename B::Scalar> to_host_data(std::span<const typename B::Scalar> data)
  {
    std::vector<typename B::Scalar> host_data(data.size());
    std::copy_n(data.data(), data.size(), host_data.data());
    return host_data;
  }

  void sync() {}

private:
  std::mt19937 rng;
  std::normal_distribution<typename B::Scalar> dist;
};
