import scipy
import tensorflow as tf

import utils


def relaxation_matrices(As, tensor=False):
    # computes the iteration matrix of the relaxation, here Gauss-Seidel is used.
    # This function is called on each block separately.
    num_As = len(As)
    grid_sizes = [A.shape[0] for A in As]
    Bs = [A.toarray() for A in As]
    for B, grid_size in zip(Bs, grid_sizes):
        B[utils.tril_indices(grid_size)[0], utils.tril_indices(grid_size)[1]] = 0.  # B is the upper part of A
    res = []
    if tensor:
        for i in range(num_As):
            res.append(tf.linalg.triangular_solve(As[i].toarray(),
                                                  -Bs[i],
                                                  lower=True))
    else:
        for i in range(num_As):
            res.append(scipy.linalg.solve_triangular(a=As[i].toarray(),
                                                     b=-Bs[i],
                                                     lower=True, unit_diagonal=False,
                                                     overwrite_b=True, debug=None, check_finite=False).astype(
                As[i].dtype))
    return res
