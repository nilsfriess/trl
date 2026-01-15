#pragma once

namespace trl {

struct EigensolverParams {
  unsigned int nev;
  unsigned int ncv;
  unsigned int max_restarts;
};

struct EigensolverResult {
  bool converged;
  unsigned int iterations;
  unsigned int n_op_apply;
};

} // namespace trl
