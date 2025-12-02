#if HAVE_SYCL_BACKEND
#include <sycl/sycl.hpp>

#include "sycl/diagonal.hh"
#endif

#include <cassert>
#include <iostream>
#include <cmath>

#if HAVE_SYCL_BACKEND
void test_diagonal_orthonormalize_sycl() {
    sycl::queue q;
    const int N = 8;
    DiagonalEVP<double, 1> evp(q, N);
    auto V = evp.create_multivector(N, 1);
    auto V0 = V.block_view(0);
    for (int i = 0; i < N; ++i) {
        V0.data[i] = 1.0;
    }
    double* r_data = new double[1]{0.0};
    trl::MatrixBlockView<double, 1> R(&q, r_data);
    evp.orthonormalize(V0, R);
    // After orthonormalization, V0 should be normalized
    double norm = 0.0;
    for (int i = 0; i < N; ++i) norm += V0.data[i] * V0.data[i];
    norm = std::sqrt(norm);
    assert(std::abs(norm - 1.0) < 1e-12);
    delete[] r_data;
}
#endif

int main() {
#if HAVE_SYCL_BACKEND
    test_diagonal_orthonormalize_sycl();
    std::cout << "test_diagonal_orthonormalize passed\n";
#endif
    return 0;
}
