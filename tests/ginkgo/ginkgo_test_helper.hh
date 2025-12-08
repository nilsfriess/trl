#pragma once

#include <trl/concepts.hh>

#include <random>
#include <vector>

#include <ginkgo/ginkgo.hpp>

template <trl::Eigenproblem EVP>
class GinkgoTestHelper {
public:
  GinkgoTestHelper()
      : dist(0, 1)
  {
  }

  void set_random(typename EVP::BlockMultivector::BlockView V)
  {
    // TODO: This only works on host executors
    for (std::size_t i = 0; i < V.rows() * V.cols(); ++i) V.data()->get_values()[i] = dist(rng);
  }

  std::vector<typename EVP::Scalar> to_host_data(typename EVP::BlockMultivector::BlockMatrix::BlockView B) const
  {
    auto host_exec = B.data()->get_executor()->get_master();
    auto host_data = gko::clone(host_exec, B.data());
    std::vector<typename EVP::Scalar> data(B.rows() * B.cols());
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = host_data->get_values()[i];
    return data;
  }

  std::vector<typename EVP::Scalar> to_host_data(const gko::array<typename EVP::Scalar>& data) { return data.copy_to_host(); }

  typename EVP::Scalar norm(typename EVP::BlockMultivector::BlockView V)
  {
    auto host_exec = V.data()->get_executor()->get_master();
    auto host_data = gko::clone(host_exec, V.data());
    typename EVP::Scalar n = 0;
    for (std::size_t i = 0; i < V.rows() * V.cols(); ++i) n += host_data->get_values()[i] * host_data->get_values()[i];
    return std::sqrt(n);
  }

private:
  std::mt19937 rng;
  std::normal_distribution<typename EVP::Scalar> dist;
};
