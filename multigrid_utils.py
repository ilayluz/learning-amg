from functools import lru_cache

import matlab.engine
import numpy as np
import pyamg
import scipy.linalg
import tensorflow as tf
from pyamg.classical import direct_interpolation
from scipy.sparse import csr_matrix

from utils import chunks, most_frequent_splitting


def frob_norm(a, power=1):
    if power == 1:
        return tf.norm(a, axis=[-2, -1])
    else:
        curr_power = a
        for i in range(power - 1):
            curr_power = a @ curr_power
        return tf.norm(curr_power, axis=[-2, -1]) ** (1 / power)


def compute_coarse_A(R, A, P):
    return R @ A @ P


def compute_coarse_As(padded_Rs, padded_As, padded_Ps):
    RAs = padded_Rs @ padded_As
    RAPs = RAs @ padded_Ps
    return RAPs


def compute_Cs(padded_As, Is, padded_Ps, padded_Rs, coarse_As_inv):
    RAs = padded_Rs @ padded_As
    coarse_A_inv_RAs = coarse_As_inv @ RAs
    P_coarse_A_inv_RAs = padded_Ps @ coarse_A_inv_RAs
    Cs = Is - P_coarse_A_inv_RAs
    return Cs


def two_grid_error_matrices(padded_As, padded_Ps, padded_Rs, padded_Ss):
    batch_size = padded_As.shape[0].value
    padded_length = padded_As.shape[1].value
    Is = tf.eye(padded_length, batch_shape=[batch_size], dtype=padded_As.dtype)
    coarse_As = compute_coarse_As(padded_Rs, padded_As, padded_Ps)
    coarse_As_inv = tf.linalg.inv(coarse_As)
    Cs = compute_Cs(padded_As, Is, padded_Ps, padded_Rs, coarse_As_inv)
    Ms = padded_Ss @ Cs @ padded_Ss
    return Ms


def two_grid_error_matrix(A, P, R, S):
    I = tf.eye(A.shape[0].value, dtype=A.dtype)
    coarse_A = compute_coarse_A(R, A, P)
    coarse_A_inv = tf.linalg.inv(coarse_A)
    C = compute_C(A, I, P, R, coarse_A_inv)
    M = S @ C @ S
    return M


def compute_C(A, I, P, R, coarse_A_inv):
    RA = R @ A
    coarse_A_inv_RA = coarse_A_inv @ RA
    P_coarse_A_inv_RA = P @ coarse_A_inv_RA
    C = I - P_coarse_A_inv_RA
    return C


def block_diag_multiply(W_conj_t, As, W):
    return W_conj_t @ As @ W


def extract_diag_blocks(block_diag_As, block_size, root_num_blocks, single_matrix=False):
    """extracts the block matrices on the diagonal"""
    if single_matrix:
        return [
            block_diag_As[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size]
            for i in range(root_num_blocks ** 2)]
    else:
        return [
            block_diag_As[:, i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size]
            for i in range(root_num_blocks ** 2)]


def block_diagonalize_A_fast(As, root_num_blocks, tensor=False):
    """Returns root_num_blocks**2 matrices that represent the block diagonalization of A"""
    if tensor:
        total_size = As.shape[1].value
    else:
        total_size = As.shape[1]
    block_size = total_size // root_num_blocks

    double_W, double_W_conj_t = create_double_W(block_size, root_num_blocks, tensor)
    block_diag_A = block_diag_multiply(double_W_conj_t, As, double_W)

    small_block_size = block_size // root_num_blocks
    blocks = extract_diag_blocks(block_diag_A, small_block_size, root_num_blocks)

    if tensor:
        return tf.stack(blocks, axis=1)
    else:
        return [csr_matrix(block) for block_list in blocks for block in block_list]


def block_diagonalize_A_single(A, root_num_blocks, tensor=False):
    """Returns root_num_blocks**2 matrices that represent the block diagonalization of A"""
    if tensor:
        total_size = A.shape[0].value
    else:
        total_size = A.shape[0]
    block_size = total_size // root_num_blocks

    double_W, double_W_conj_t = create_double_W(block_size, root_num_blocks, tensor)
    block_diag_A = block_diag_multiply(double_W_conj_t, A, double_W)

    small_block_size = block_size // root_num_blocks
    blocks = extract_diag_blocks(block_diag_A, small_block_size, root_num_blocks, single_matrix=True)
    blocks = blocks[1:]  # ignore zero mode block

    if tensor:
        return tf.stack(blocks, axis=0)
    else:
        return [csr_matrix(block) for block_list in blocks for block in block_list]


def block_diagonalize_A(A, root_num_blocks):
    """Returns root_num_blocks**2 matrices that represent the block diagonalization of A"""
    block_size = A.shape[0] // root_num_blocks

    # block-diagonalize each of the blocks in the first row of blocks (no need to block-diagonalize all blocks,
    # because A is block-circulant)
    small_W, small_W_conj_t = create_W_matrix(block_size // root_num_blocks, root_num_blocks)
    small_diagonalized_blocks = []
    for i in range(root_num_blocks):
        small_block = A[:block_size, i * block_size:(i + 1) * block_size]
        small_diagonalized_block = small_W_conj_t @ small_block @ small_W
        small_diagonalized_blocks.append(small_diagonalized_block)

    # arrange the block-diagonalized blocks into a block-circulant matrix
    block_list = []
    for shift in range(root_num_blocks):
        shifted_list = np.roll(small_diagonalized_blocks, shift, axis=0)
        block_list.append(list(shifted_list))
    small_block_diagonalized_A = np.block(block_list)

    # block-diagonalize the block-circulant matrix, extract the resulting blocks and stack them
    double_block_diagonalized_A = block_diagonalize_1d_circulant(small_block_diagonalized_A, root_num_blocks)
    small_blocks = [
        double_block_diagonalized_A[b, i:i + block_size // root_num_blocks, i:i + block_size // root_num_blocks]
        for i in range(0, A.shape[0] // root_num_blocks, block_size // root_num_blocks)
        for b in range(root_num_blocks)]
    return np.stack(small_blocks)


def pad_P(P, coarse_nodes):
    total_size = P.shape[0].value
    zero_column = tf.zeros([total_size], dtype=tf.float64)
    P_cols = tf.unstack(P, axis=1)
    full_P_cols = []
    curr_P_col = 0
    is_coarse = np.in1d(range(total_size), coarse_nodes, assume_unique=True)
    for col_index in range(total_size):
        if is_coarse[col_index]:
            column = P_cols[curr_P_col]
            curr_P_col += 1
        else:
            column = zero_column
        full_P_cols.append(column)

    full_P = tf.transpose(tf.stack(full_P_cols))
    full_P = tf.cast(full_P, tf.complex128)
    return full_P


def block_diagonalize_P(P, root_num_blocks, coarse_nodes):
    """
    Returns root_num_blocks**2 matrices that represent the block diagonalization of P
    Only works on block-periodic prolongation matrices
    """
    total_size = P.shape[0].value
    block_size = total_size // root_num_blocks

    # we build the padded P matrix column by column, I couldn't find a more efficient way
    full_P = pad_P(P, coarse_nodes)

    double_W, double_W_conj_t = create_double_W(block_size, root_num_blocks, True)
    block_diag_full_P = block_diag_multiply(double_W_conj_t, full_P, double_W)

    small_block_size = block_size // root_num_blocks
    blocks = extract_diag_blocks(block_diag_full_P, small_block_size, root_num_blocks, single_matrix=True)
    blocks = blocks[1:]  # ignore zero mode block

    block_coarse_nodes = coarse_nodes[:len(coarse_nodes) // root_num_blocks**2]
    blocks = [tf.gather(block, block_coarse_nodes, axis=1) for block in blocks]

    return blocks


def block_diagonalize_1d_circulant(A, root_num_blocks):
    """
    Returns root_num_blocks matrices that represent the block diagonalization of A
    We apply this function recursively to block-diagonalize 2d-block-circulant matrices
    Refer to docs/block_fourier_analysis.pdf for notation and details
    """
    total_size = A.shape[0]
    block_size = total_size // root_num_blocks
    W, W_conj_t = create_W_matrix(block_size, root_num_blocks)
    block_diagonal_matrix = W_conj_t @ A @ W

    # extract the block matrices on the diagonal
    blocks = [block_diagonal_matrix[i:i + block_size, i:i + block_size] for i in range(0, total_size, block_size)]
    return np.stack(blocks)


@lru_cache(maxsize=None)
def create_W_matrix(block_size, root_num_blocks, tensor=False):
    """
    Returns a matrix that block-diagonalizes a block-circulant matrix
    Refer to docs/block_fourier_analysis.pdf for notation and details
    """
    total_size = block_size * root_num_blocks
    dft_matrix = scipy.linalg.dft(total_size)
    dft_matrix_first_b_columns = dft_matrix[:, :root_num_blocks]

    columns = []
    for i in range(root_num_blocks):
        for k in range(block_size):
            col_mask = np.ones(total_size, np.bool)
            col_mask[k:total_size:block_size] = 0
            column = np.copy(dft_matrix_first_b_columns[:, i])
            column[col_mask] = 0
            columns.append(column)
    W = np.stack(columns, axis=1)
    W /= np.sqrt(root_num_blocks)
    W_conj_t = W.conj().T

    if tensor:
        W, W_conj_t = tf.convert_to_tensor(W), tf.convert_to_tensor(W_conj_t)
    return W, W_conj_t


@lru_cache(maxsize=None)
def create_double_W(block_size, root_num_blocks, tensor=False):
    big_W, _ = create_W_matrix(block_size, root_num_blocks)
    small_W, _ = create_W_matrix(block_size // root_num_blocks, root_num_blocks)
    small_W_block = scipy.linalg.block_diag(*[small_W] * root_num_blocks)
    double_W = small_W_block @ big_W
    double_W_conj_t = double_W.conj().T

    if tensor:
        double_W, double_W_conj_t = tf.convert_to_tensor(double_W), tf.convert_to_tensor(double_W_conj_t)
    return double_W, double_W_conj_t


def test_create_W_matrix():
    """Check if W matrix is unitary"""
    W, W_conj_T = create_W_matrix(3, 4)
    I = W_conj_T @ W
    print(np.all(np.isclose(I, np.eye(3 * 4))))


def test_block_diagonalize_1d_circulant():
    """Check if eigenvalues of block matrices are the same as eigenvalues of original block-circulant matrix"""
    matlab_engine = matlab.engine.start_matlab()
    matlab_engine.eval('rng(1)')  # fix random seed for reproducibility

    def generate_A_delaunay_block_periodic_lognormal(num_unknowns_per_block, root_num_blocks, matlab_engine):
        """Poisson equation on triangular mesh, with lognormal coefficients, and block periodic boundary conditions"""
        # points are correct only for 3x3 number of blocks
        A_matlab, points_matlab = matlab_engine.block_periodic_delaunay(num_unknowns_per_block, root_num_blocks,
                                                                        nargout=2)
        A_numpy = np.array(A_matlab._data).reshape(A_matlab.size, order='F')
        points_numpy = np.array(points_matlab._data).reshape(points_matlab.size, order='F')
        return csr_matrix(A_numpy), points_numpy

    A, _ = generate_A_delaunay_block_periodic_lognormal(3, 4, matlab_engine)
    A = A.toarray()
    blocks = block_diagonalize_1d_circulant(A, 4)
    A_eigs = np.sort(np.linalg.eigvals(A))
    block_eigs = np.sort(np.linalg.eigvals(blocks).flatten())
    print(np.all(np.isclose(A_eigs, block_eigs)))


def test_block_diagonalize_A():
    """Check if eigenvalues of block matrices are the same as eigenvalues of original block-circulant matrix"""
    matlab_engine = matlab.engine.start_matlab()
    matlab_engine.eval('rng(1)')  # fix random seed for reproducibility

    def generate_A_delaunay_block_periodic_lognormal(num_unknowns_per_block, root_num_blocks, matlab_engine):
        """Poisson equation on triangular mesh, with lognormal coefficients, and block periodic boundary conditions"""
        # points are correct only for 3x3 number of blocks
        A_matlab, points_matlab = matlab_engine.block_periodic_delaunay(num_unknowns_per_block, root_num_blocks,
                                                                        nargout=2)
        A_numpy = np.array(A_matlab._data).reshape(A_matlab.size, order='F')
        points_numpy = np.array(points_matlab._data).reshape(points_matlab.size, order='F')
        return csr_matrix(A_numpy), points_numpy

    A, _ = generate_A_delaunay_block_periodic_lognormal(15, 4, matlab_engine)
    A = A.toarray()
    blocks = block_diagonalize_A(A, 4)

    # check if eigenvalues are identical
    A_eigs = np.sort(np.linalg.eigvals(A))
    block_eigs = np.sort(np.linalg.eigvals(blocks).flatten())
    print(np.all(np.isclose(A_eigs, block_eigs)))


def test_block_diagonalize_A_fast():
    """Check if eigenvalues of block matrices are the same as eigenvalues of original block-circulant matrix"""
    matlab_engine = matlab.engine.start_matlab()
    matlab_engine.eval('rng(1)')  # fix random seed for reproducibility

    def generate_A_delaunay_block_periodic_lognormal(num_unknowns_per_block, root_num_blocks, matlab_engine):
        """Poisson equation on triangular mesh, with lognormal coefficients, and block periodic boundary conditions"""
        # points are correct only for 3x3 number of blocks
        A_matlab = matlab_engine.block_periodic_delaunay(num_unknowns_per_block, root_num_blocks,
                                                                        nargout=1)
        A_numpy = np.array(A_matlab._data).reshape(A_matlab.size, order='F')
        return csr_matrix(A_numpy)

    batch_size = 32
    As = [generate_A_delaunay_block_periodic_lognormal(5, 3, matlab_engine) for i in range(batch_size)]
    As = [A.toarray() for A in As]
    As = tf.stack(As)
    As = tf.cast(As, dtype=tf.complex128)
    blocks = block_diagonalize_A_fast(As, 3, tensor=True).numpy()

    # check if eigenvalues are identical
    A_eigs = np.sort(np.linalg.eigvals(As.numpy()).flatten())
    block_eigs = np.sort(np.linalg.eigvals(blocks).flatten())
    print(np.all(np.isclose(A_eigs, block_eigs)))


def test_block_diagonalize_P():
    """Check if eigenvalues of block matrices are the same as eigenvalues of original block-circulant matrix"""
    matlab_engine = matlab.engine.start_matlab()
    matlab_engine.eval('rng(1)')  # fix random seed for reproducibility

    def generate_A_delaunay_block_periodic_lognormal(num_unknowns_per_block, root_num_blocks, matlab_engine):
        """Poisson equation on triangular mesh, with lognormal coefficients, and block periodic boundary conditions"""
        # points are correct only for 3x3 number of blocks
        A_matlab = matlab_engine.block_periodic_delaunay(num_unknowns_per_block, root_num_blocks,
                                                                        nargout=1)
        A_numpy = np.array(A_matlab._data).reshape(A_matlab.size, order='F')
        return csr_matrix(A_numpy)

    num_unknowns_per_block = 64
    root_num_blocks = 3
    A = generate_A_delaunay_block_periodic_lognormal(num_unknowns_per_block, root_num_blocks, matlab_engine)
    # A = A + 0.1 * scipy.sparse.diags(np.ones(num_unknowns_per_block * root_num_blocks**2))

    orig_solver = pyamg.ruge_stuben_solver(A, max_levels=2, max_coarse=1, CF='CLJP', keep=True)
    orig_splitting = orig_solver.levels[0].splitting
    block_splitting = list(chunks(orig_splitting, num_unknowns_per_block))
    common_block_splitting = most_frequent_splitting(block_splitting)
    repeated_splitting = np.tile(common_block_splitting, root_num_blocks ** 2)

    # we recompute the Ruge-Stuben prolongation matrix with the modified splitting, and the original strength
    # matrix. We assume the strength matrix is block-circulant (because A is block-circulant)
    C = orig_solver.levels[0].C
    P = direct_interpolation(A, C, repeated_splitting)
    P = tf.convert_to_tensor(P.toarray(), dtype=tf.float64)

    P_blocks = block_diagonalize_P(P, root_num_blocks, repeated_splitting.nonzero())
    P_blocks = P_blocks.numpy()

    A_c = P.numpy().T @ A.toarray() @ P.numpy()
    # double_W, double_W_conj_t = create_double_W(num_unknowns_per_block * root_num_blocks, root_num_blocks)
    # A_c_full_block_diag = double_W_conj_t @ A_c_full @ double_W

    # P_full_block_diag = double_W_conj_t @ full_P @ double_W
    # A_block_diag = double_W_conj_t @ A.toarray() @ double_W
    # A_c_full_block_diag_2 = P_full_block_diag.conj().T @ A_block_diag @ P_full_block_diag

    tf_A = tf.cast(tf.stack([A.toarray()]), tf.complex128)
    A_blocks = block_diagonalize_A_fast(tf_A, root_num_blocks, True).numpy()[0][1:]  # ignore the first zero mode block

    def relaxation_matrices(As, w=0.8):
        I = np.eye(As[0].shape[0])
        res = [I - w * np.diag(1 / (np.diag(A))) @ A for A in As]

        # computes the iteration matrix of the relaxation, here Gauss-Seidel is used.
        # This function is called on each block seperately.
        # num_As = len(As)
        # grid_sizes = [A.shape[0] for A in As]
        # Bs = [A.copy() for A in As]
        # for B, grid_size in zip(Bs, grid_sizes):
        #     B[np.tril_indices(grid_size, 0)[0], np.tril_indices(grid_size, 0)[1]] = 0.  # B is the upper part of A
        # res = []
        # for i in tqdm(range(num_As)):  # range(A.shape[0] // batch_size):
        #     res.append(scipy.linalg.solve_triangular(a=As[i],
        #                                              b=-Bs[i],
        #                                              lower=True, unit_diagonal=False,
        #                                              overwrite_b=False, debug=None, check_finite=True).astype(
        #         np.float64))
        return res

    S = relaxation_matrices([A.toarray()])[0]
    S_blocks = relaxation_matrices(A_blocks)

    # A_c = P.numpy().T @ A.toarray() @ P.numpy()
    A_c_blocks = P_blocks.transpose([0, 2, 1]).conj() @ A_blocks @ P_blocks

    A = A.toarray()
    C = np.eye(A.shape[0]) - P.numpy() @ np.linalg.inv(A_c) @ P.numpy().T @ A
    M = S @ C @ S

    I = np.eye(A_blocks[0].shape[0])
    C_blocks = [I - P_block @ np.linalg.inv(A_c_block) @ P_block.conj().T @ A_block
                for (P_block, A_c_block, A_block) in zip(P_blocks, A_c_blocks, A_blocks)]
    M_blocks = [S_block @ C_block @ S_block for (S_block, C_block) in zip(S_blocks, C_blocks)]

    # # extract only elements that correspond to coarse nodes
    # A_c_blocks = A_c_blocks[:, common_block_splitting.nonzero()[0][:, None], common_block_splitting.nonzero()[0]]


    A_c_block_eigs = np.sort(np.linalg.eigvals(A_c_blocks).flatten())
    A_c_eigs = np.sort(np.linalg.eigvals(A_c))

    C_block_eigs = np.sort(np.linalg.eigvals(C_blocks).flatten())
    C_eigs = np.sort(np.linalg.eigvals(C))


    M_block_eigs = np.sort(np.linalg.eigvals(M_blocks).flatten())
    M_eigs = np.sort(np.linalg.eigvals(M))

    pass
