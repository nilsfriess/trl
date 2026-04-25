#include "../csrtest_common.hh"
#include "generalised_csrevp.hh"
#include "trl/impl/sycl/profiling.hh"

#include <sycl/sycl.hpp>
#include <vector>

constexpr unsigned int BLOCKSIZE = 4;

int main(int argc, char* argv[])
{
  using Scalar = double;

  sycl::queue queue{{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
  using EVP = GeneralisedCSREVP<Scalar, BLOCKSIZE>;

  const int result = run_generalised_csrtest<BLOCKSIZE>(
      argc, argv, [&](const std::string& matrix_A, const std::string& matrix_B) { return std::make_shared<EVP>(queue, matrix_A, matrix_B); },
      [&]() { std::cout << "SYCL device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n\n"; },
      [&](auto V0, auto& rng, auto& dist) {
        const std::size_t n = V0.rows() * V0.cols();
        std::vector<Scalar> v0_host(n);
        std::generate_n(v0_host.begin(), n, [&]() { return static_cast<Scalar>(dist(rng)); });
        queue.memcpy(V0.data, v0_host.data(), n * sizeof(Scalar)).wait();
      },
      [&]() { queue.wait(); });

  trl::sycl::SyclProfiler::get().report();
  return result;
}
