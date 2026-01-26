#pragma once

#include <trl/concepts.hh>

#include <random>
#include <vector>

#include <ginkgo/ginkgo.hpp>
#include <hip/hip_runtime.h>

#include "../examples/ginkgo/hiputils.hh"

template <trl::Eigenproblem EVP>
class GinkgoTestHelper {
public:
  GinkgoTestHelper()
      : dist(0, 1)
  {
  }

  void set_random(typename EVP::BlockMultivector::BlockView V)
  {
    auto exec = V.data()->get_executor();
    
    // Generate random values on the host
    std::vector<typename EVP::Scalar> host_values(V.rows() * V.cols());
    for (std::size_t i = 0; i < host_values.size(); ++i) {
      host_values[i] = dist(rng);
    }
    
    std::cout << "set_random: setting " << V.rows() << "x" << V.cols() << " matrix" << std::endl;
    std::cout << "set_random: V.data() pointer = " << V.data() << std::endl;
    std::cout << "set_random: V.data()->get_values() = " << V.data()->get_values() << std::endl;
    
    // Copy to device 
    if (std::dynamic_pointer_cast<const gko::HipExecutor>(exec)) {
      // Use HIP directly for device executors
      auto err = hipMemcpy(V.data()->get_values(), host_values.data(), host_values.size() * sizeof(typename EVP::Scalar), hipMemcpyHostToDevice);
      if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipMemcpy failed");
      }
      HIP_CHECK(hipDeviceSynchronize());
      
      // Verify immediately
      std::vector<typename EVP::Scalar> verify(host_values.size());
      err = hipMemcpy(verify.data(), V.data()->get_values(), verify.size() * sizeof(typename EVP::Scalar), hipMemcpyDeviceToHost);
      double norm = 0;
      for (auto v : verify) norm += v * v;
      std::cout << "set_random: Verified ||V||^2 = " << norm << std::endl;
    } else {
      // For CPU executors, direct assignment works
      for (std::size_t i = 0; i < V.rows(); ++i) {
        for (std::size_t j = 0; j < V.cols(); ++j) {
          V.data()->at(i, j) = host_values[i * V.cols() + j];
        }
      }
    }
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
