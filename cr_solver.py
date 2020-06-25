from warnings import warn

import numpy as np
import scipy.sparse as sparse
from pyamg.classical import split
from pyamg.classical.cr import CR
from pyamg.multilevel import multilevel_solver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.classical.interpolate import distance_two_interpolation, direct_interpolation, standard_interpolation
from pyamg.strength import classical_strength_of_connection
from scipy.sparse import csr_matrix, isspmatrix_csr, SparseEfficiencyWarning


def cr_solver(A,
              CF='CR', l=40, maxp=40000, theta_a=0.25*0,
              presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
              postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
              max_levels=10, max_coarse=10, keep=False, **kwargs):
    """Create a multilevel solver using Classical AMG (Ruge-Stuben AMG).

    Parameters
    ----------
    A : csr_matrix
        Square matrix in CSR format
    CF : string
        Method used for coarse grid selection (C/F splitting)
        Supported methods are RS, PMIS, PMISc, CLJP, CLJPc, and CR.
    presmoother : string or dict
        Method used for presmoothing at each level.  Method-specific parameters
        may be passed in using a tuple, e.g.
        presmoother=('gauss_seidel',{'sweep':'symmetric}), the default.
    postsmoother : string or dict
        Postsmoothing method with the same usage as presmoother
    max_levels: integer
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: integer
        Maximum number of variables permitted on the coarse grid.
    keep: bool
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  For example, if True, then strength of connection (C) and
        tentative prolongation (T) are kept.

    Returns
    -------
    ml : multilevel_solver
        Multigrid hierarchy of matrices and prolongation operators

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg import ruge_stuben_solver
    >>> A = poisson((10,),format='csr')
    >>> ml = ruge_stuben_solver(A,max_coarse=3)

    Notes
    -----
    "coarse_solver" is an optional argument and is the solver used at the
    coarsest grid.  The default is a pseudo-inverse.  Most simply,
    coarse_solver can be one of ['splu', 'lu', 'cholesky, 'pinv',
    'gauss_seidel', ... ].  Additionally, coarse_solver may be a tuple
    (fn, args), where fn is a string such as ['splu', 'lu', ...] or a callable
    function, and args is a dictionary of arguments to be passed to fn.
    See [2001TrOoSc]_ for additional details.


    References
    ----------
    .. [2001TrOoSc] Trottenberg, U., Oosterlee, C. W., and Schuller, A.,
       "Multigrid" San Diego: Academic Press, 2001.  Appendix A

    See Also
    --------
    aggregation.smoothed_aggregation_solver, multilevel_solver,
    aggregation.rootnode_solver

    """
    levels = [multilevel_solver.level()]

    # convert A to csr
    if not isspmatrix_csr(A):
        try:
            A = csr_matrix(A)
            warn("Implicit conversion of A to CSR",
                 SparseEfficiencyWarning)
        except BaseException:
            raise TypeError('Argument A must have type csr_matrix, \
                             or be convertible to csr_matrix')
    # preprocess A
    A = A.asfptype()
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels[-1].A = A

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        extend_hierarchy(levels, CF, l, maxp, theta_a, keep)

    ml = multilevel_solver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


def extend_hierarchy(levels, CF, l, maxp, theta_a, keep):
    """Extend the multigrid hierarchy."""

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    A = levels[-1].A

    # Generate the C/F splitting
    fn, kwargs = unpack_arg(CF)
    if fn == 'CR':
        splitting = CR(A, **kwargs)
    else:
        raise ValueError('unknown C/F splitting method (%s)' % CF)

    # rs_C = classical_strength_of_connection(A, theta=0.25)
    # rs_splitting = split.RS(rs_C)
    # rs_P = direct_interpolation(A.copy(), rs_C.copy(), rs_splitting.copy())
    #
    # rs_P_sparsity = rs_P.copy()
    # rs_P_sparsity.data[:] = 1
    #
    # rs_fine = np.where(rs_splitting == 0)[0]
    # rs_coarse = np.where(rs_splitting == 1)[0]
    # rs_A_fc = A[rs_fine][:, rs_coarse]
    # rs_W = rs_P[rs_fine]
    # my_rs_P, my_rs_W = my_direct_interpolation(rs_A_fc, A, rs_W, rs_coarse, rs_fine)
    #
    # my_rs_P_sparsity = my_rs_P.copy()
    # my_rs_P_sparsity.data[:] = 1
    #
    # rs_A_sparsity = A[:, rs_coarse].copy()
    # rs_A_sparsity.data[:] = 1

    # Generate the interpolation matrix that maps from the coarse-grid to the
    # fine-grid
    P = truncation_interpolation(A, splitting, l, maxp, theta_a)
    # P = optimal_interpolation(A, splitting)
    # P = rs_P

    # Generate the restriction matrix that maps from the fine-grid to the
    # coarse-grid
    R = P.T.tocsr()

    # Store relevant information for this level
    if keep:
        levels[-1].splitting = splitting  # C/F splitting

    levels[-1].P = P  # prolongation operator
    levels[-1].R = R  # restriction operator

    levels.append(multilevel_solver.level())

    # Form next level through Galerkin product
    A = R * A * P
    levels[-1].A = A


def optimal_interpolation(A, splitting):
    fine = np.where(splitting == 0)[0]
    coarse = np.where(splitting == 1)[0]

    A_ff = A[fine][:, fine]
    A_fc = A[fine][:, coarse]

    W = -sparse.linalg.inv(A_ff) @ A_fc

    np_W = W.toarray()
    P = np.zeros(A.shape)
    for i in range(A_fc.shape[1]):
        P[fine, coarse[i]] = np_W[:, i]

    np.fill_diagonal(P, 1)
    P = P[:, coarse]
    P = csr_matrix(P)
    return P


def my_direct_interpolation(A_fc, A, sparsity, coarse, fine):
    sparsity = sparsity.copy()
    sparsity.data[:] = 1.0
    sparsity = sparsity.multiply(A_fc)

    A_zerodiag = A - sparse.diags(A.diagonal())
    # A_zerodiag = A
    A_rowsums = np.array(A_zerodiag.sum(axis=1))[:, 0]
    sparsity_rowsums = np.array(sparsity.sum(axis=1))[:, 0]

    W = -A_fc.multiply(A_rowsums[fine, None]) / A.diagonal()[fine, None] / sparsity_rowsums[:, None]

    np_W = np.array(W)
    n = A_fc.shape[0] + A_fc.shape[1]
    P_square = np.zeros((n, n))
    for i in range(W.shape[1]):
        P_square[fine, coarse[i]] = np_W[:, i]

    np.fill_diagonal(P_square, 1)
    P_square = csr_matrix(P_square)

    P = P_square[:, coarse]
    return P, csr_matrix(W)


# from "Compatible Relaxation and Coarsening in Algebraic Multigrid" (2009)
def truncation_interpolation(A, splitting, l, maxp, theta_a):
    # eq. 3.2
    fine = np.where(splitting == 0)[0]
    coarse = np.where(splitting == 1)[0]

    A_ff = A[fine][:, fine]
    A_fc = A[fine][:, coarse]

    D_ff = sparse.diags(A_ff.diagonal()).tocsr()
    D_ffinv = D_ff.power(-1)

    # eq. 4.8
    omega = 1 / gershgorin_bound(D_ffinv @ A_ff)

    # eq. 4.7
    W = -weighted_jacobi_cr(omega, D_ffinv, A_fc, A_ff, l)

    # W_star = -sparse.linalg.inv(A_ff) @ A_fc

    W = keep_largest_per_row(W, maxp)
    # W = keep_largest_per_row(W_star, maxp)

    # eq. 4.9
    sparsity = keep_thres_per_row(W, theta_a)
    # sparsity = W_star
    sparsity.data = np.ones_like(sparsity.data)

    # my_P = my_direct_interpolation(A_fc, A, sparsity, coarse, fine)

    # eq. 4.10
    # TODO: implement more efficient indexing by passing to matlab
    np_sparsity = sparsity.toarray()
    P_sparsity = np.zeros(A.shape)
    for i in range(sparsity.shape[1]):
        P_sparsity[fine, coarse[i]] = np_sparsity[:, i]

    np.fill_diagonal(P_sparsity, 0)
    P_sparsity = csr_matrix(P_sparsity)

    # P = distance_two_interpolation(A.copy(), P_sparsity.copy(), splitting.copy())
    P = direct_interpolation(A.copy(), P_sparsity.copy(), splitting.copy())
    # P = standard_interpolation(A.copy(), P_sparsity.copy(), splitting.copy())

    return P
    # return my_P


def gershgorin_bound(M):
    return abs(M).sum(axis=1).max()


def weighted_jacobi_cr(omega, D_ffinv, A_fc, A_ff, l):
    W = csr_matrix((A_fc.shape[0], A_fc.shape[1]))
    for _ in range(l):
        W = W + omega * D_ffinv @ (A_fc - A_ff @ W)
    W.eliminate_zeros()
    return W


def keep_largest_per_row(M, maxp):
    nrows = M.shape[0]
    for i in range(nrows):
        # Get the row slice, not a copy, only the non zero elements
        row_array = M.data[M.indptr[i]: M.indptr[i + 1]]
        if row_array.shape[0] <= maxp:
            # Not more than maxp elements
            continue

        # only take the maxp last elements in the sorted indices
        row_array[np.argsort(row_array)[:-maxp]] = 0
    M.eliminate_zeros()
    return M


def keep_thres_per_row(M, theta_a):
    nrows = M.shape[0]
    for i in range(nrows):
        # Get the row slice, not a copy, only the non zero elements
        row_array = M.data[M.indptr[i]: M.indptr[i + 1]]

        threshold = theta_a * max(abs(row_array))
        row_array[np.where(abs(row_array) <= threshold)] = 0
    M.eliminate_zeros()
    return M
