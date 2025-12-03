#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// clang-format off
#define TODO(msg) do { \
    /* Emacs compilation buffer format: file:line: message */ \
    std::cerr << __FILE__ << ":" << __LINE__ << ": TODO in " << __func__ << ": " << msg << std::endl; \
    std::exit(EXIT_FAILURE); \
} while(0)
// clang-format on

#include "trl/concepts.hh"
#include "trl/eigensolvers/lanczos.hh"
#include "trl/eigensolvers/params.hh"
#include "trl/impl/gingko/multivector.hh"

#include <ginkgo/ginkgo.hpp>

template <class T, unsigned int bs>
class DiagonalEigenproblem {
public:
  using Scalar = T;
  using BlockMultivector = trl::ginkgo::BlockMultivector<T, bs>;
  using BlockView = typename BlockMultivector::BlockView;
  using BlockMatrix = typename BlockMultivector::BlockMatrix;
  using MatrixBlockView = typename BlockMatrix::BlockView;
  static constexpr unsigned int blocksize = bs;

  DiagonalEigenproblem(std::shared_ptr<const gko::Executor> exec, std::size_t N)
      : exec_(std::move(exec))
      , N_(N)
  {
  }

  void apply(BlockView x, BlockView y) { TODO("Not implemented"); }

  void dot(BlockView x, BlockView y, MatrixBlockView B) { TODO("Not implemented"); }

  void orthonormalize(BlockView x, MatrixBlockView B) { TODO("Not implemented"); }

  std::size_t size() const { return N_; }

  BlockMultivector create_multivector(std::size_t rows, std::size_t cols) const { return BlockMultivector(exec_, rows, cols); }

  BlockMatrix create_blockmatrix(std::size_t block_rows, std::size_t block_cols) const { return BlockMatrix(exec_, block_rows, block_cols); }

  void solve_small_dense(const BlockMatrix& B) { TODO("Not implemented"); }

  const gko::array<T>& get_current_eigenvalues() const { TODO("Not implemented"); }

  const gko::array<T>& get_eigenvalues_block(std::size_t block) { TODO("Not implemented"); }

  const gko::array<T>& get_temp_vector(std::size_t min_size) { TODO("Not implemented"); }

  const std::vector<T>& two_norm_on_host(MatrixBlockView B) { TODO("Not implemented"); }

  const BlockMatrix& get_current_eigenvectors() const { TODO("Not implemented"); }

private:
  std::shared_ptr<const gko::Executor> exec_;
  std::size_t N_;
};

int main(int argc, char* argv[])
{
  using Scalar = double;
  constexpr unsigned int bs = 1;
  std::size_t N = 1024;
  std::size_t nev = 8;

  using BMV = trl::ginkgo::BlockMultivector<Scalar, bs>;
  using EVP = DiagonalEigenproblem<Scalar, bs>;
  static_assert(trl::BlockMultiVector<BMV>);
  static_assert(trl::Eigenproblem<EVP>);

  auto exec = gko::OmpExecutor::create();

  auto evp = std::make_shared<EVP>(exec, N);

  trl::EigensolverParams params{
      .nev = 9,
      .ncv = 32,
  };

  trl::BlockLanczos lanczos(evp, params);
  lanczos.solve();

  // // Some shortcuts
  // using ValueType = double;
  // using IndexType = int;

  // using array = gko::array<ValueType>;

  // using vec = gko::matrix::Dense<ValueType>;
  // using mtx = gko::matrix::Csr<ValueType, IndexType>;
  // using cg = gko::solver::Cg<ValueType>;
  // using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;

  // // Print version information
  // std::cout << gko::version_info::get() << std::endl;

  // if (argc == 2 && (std::string(argv[1]) == "--help")) {
  //     std::cerr << "Usage: " << argv[0]
  //               << " [executor] [DISCRETIZATION_POINTS]" << std::endl;
  //     std::exit(-1);
  // }

  // // Get number of discretization points
  // const unsigned int discretization_points =
  //     argc >= 3 ? std::atoi(argv[2]) : 100;

  // // Get the executor string
  // const auto executor_string = argc >= 2 ? argv[1] : "reference";

  // // Figure out where to run the code
  // std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
  //     exec_map{
  //         {"omp", [] { return gko::OmpExecutor::create(); }},
  //         {"cuda",
  //          [] {
  //              return gko::CudaExecutor::create(0,
  //                                               gko::OmpExecutor::create());
  //          }},
  //         {"hip",
  //          [] {
  //              return gko::HipExecutor::create(0, gko::OmpExecutor::create());
  //          }},
  //         {"dpcpp",
  //          [] {
  //              return gko::DpcppExecutor::create(0,
  //                                                gko::OmpExecutor::create());
  //          }},
  //         {"reference", [] { return gko::ReferenceExecutor::create(); }}};

  // // executor where Ginkgo will perform the computation
  // const auto exec = exec_map.at(executor_string)();  // throws if not valid
  // // executor used by the application
  // const auto app_exec = exec->get_master();

  // // Set up the problem: define the exact solution, the right hand side and
  // // the Dirichlet boundary condition.
  // auto correct_u = [](ValueType x) { return x * x * x; };
  // auto f = [](ValueType x) { return ValueType(6) * x; };
  // auto u0 = correct_u(0);
  // auto u1 = correct_u(1);

  // // initialize matrix and vectors
  // auto matrix = mtx::create(app_exec, gko::dim<2>(discretization_points),
  //                           3 * discretization_points - 2);
  // generate_stencil_matrix(matrix.get());
  // auto rhs = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
  // generate_rhs(f, u0, u1, rhs.get());
  // auto u = vec::create(app_exec, gko::dim<2>(discretization_points, 1));
  // for (int i = 0; i < u->get_size()[0]; ++i) {
  //     u->get_values()[i] = 0.0;
  // }

  // const gko::remove_complex<ValueType> reduction_factor = 1e-7;
  // // Generate solver and solve the system
  // cg::build()
  //     .with_criteria(
  //         gko::stop::Iteration::build().with_max_iters(discretization_points),
  //         gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(
  //             reduction_factor))
  //     .with_preconditioner(bj::build())
  //     .on(exec)
  //     ->generate(clone(exec, matrix))  // copy the matrix to the executor
  //     ->apply(rhs, u);

  // // Uncomment to print the solution
  // // print_solution<ValueType>(u0, u1, u.get());
  // std::cout << "Solve complete.\nThe average relative error is "
  //           << calculate_error(discretization_points, u.get(), correct_u) /
  //                  static_cast<gko::remove_complex<ValueType>>(
  //                      discretization_points)
  //           << std::endl;
}
