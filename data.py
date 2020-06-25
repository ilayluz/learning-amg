import math
from collections import defaultdict
from functools import lru_cache

import matlab
import meshpy.triangle as triangle
import numpy as np
import pyamg
import scipy
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.spatial.qhull import Delaunay
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


def generate_A(size, dist, block_periodic, root_num_blocks, add_diag=False, matlab_engine=None):
    if dist is 'lognormal_laplacian':
        A = generate_A_delaunay_dirichlet_lognormal(size, matlab_engine=matlab_engine)
    elif dist is 'lognormal_laplacian_periodic':
        if block_periodic:
            A, _ = generate_A_delaunay_block_periodic_lognormal(size, root_num_blocks, matlab_engine)
        else:
            A = generate_A_delaunay_periodic_lognormal(size, uniform=False)
    elif dist is 'lognormal_complex_fem':
        A = A_dirichlet_finite_element_quality(size, matlab_engine, hole=True)
    elif dist is 'spectral_clustering':
        A = generate_A_spec_cluster(size, add_diag=add_diag)
    elif dist is 'poisson':
        grid_size = int(np.sqrt(size))
        A = pyamg.gallery.poisson((grid_size, grid_size), type='FE')
    elif dist is 'aniso':
        grid_size = int(np.sqrt(size))
        # stencil = pyamg.gallery.diffusion_stencil_2d(epsilon=0.01, type='FE')
        stencil = pyamg.gallery.diffusion_stencil_2d(epsilon=0.01, theta=np.pi / 3, type='FE')
        A = pyamg.gallery.stencil_grid(stencil, (grid_size, grid_size), format='csr')
    elif dist is 'example':
        A = pyamg.gallery.load_example(size)['A']
    return A


def drop_zero_row_col_matlab(A, matlab_engine):
    size = A.shape[0]
    A_coo = A.tocoo()
    A_rows = matlab.double((A_coo.row + 1))
    A_cols = matlab.double((A_coo.col + 1))
    A_values = matlab.double(A_coo.data)
    rows, cols, values = matlab_engine.drop_zero_row_col(A_rows, A_cols, A_values, size, nargout=3)
    rows = np.array(rows._data).reshape(rows.size, order='F') - 1
    cols = np.array(cols._data).reshape(cols.size, order='F') - 1
    values = np.array(values._data).reshape(values.size, order='F')
    rows, cols, values = rows.T[0], cols.T[0], values.T[0]
    rows, cols = rows.astype(np.int), cols.astype(np.int)
    return csr_matrix((values, (rows, cols)))


def drop_zero_row_col(A):
    # https://stackoverflow.com/a/35905815
    return A[A.getnnz(1) > 0][:, A.getnnz(0) > 0]


def generate_A_delaunay_dirichlet_lognormal(num_points, constant_coefficients=False, uniform=False,
                                            matlab_engine=None):
    """
    Poisson equation on triangular mesh, with lognormal coefficients
    We create a triangulation of random points on the square from (-1,-1) to (2,2),
    and we look only at points that lie inside the unit square. Each point that has an edge to a point
    outside the unit square, we designate as a boundary
    the total number of points in the grid, including boundaries, is num_points
    the number of unknowns is the number of points minus the number of boundaries, which is variable
    """
    rand_points = np.random.uniform([-1, -1], [2, 2], [num_points * 3 ** 2, 2])

    # remove points that lie inside the unit square
    unit_square_indices = np.where(
        (rand_points[:, 0] >= 0) &
        (rand_points[:, 0] <= 1) &
        (rand_points[:, 1] >= 0) &
        (rand_points[:, 1] <= 1)
    )[0]

    rand_points = np.delete(rand_points, unit_square_indices, axis=0)

    # add back exactly num_unknowns points to the unit square
    rand_points_unit_square = np.random.uniform([0, 0], [1, 1], [num_points, 2])
    rand_points = np.concatenate([rand_points_unit_square, rand_points])

    tri = Delaunay(rand_points)
    # vertex_neighbor_vertices is used to get neighbors:
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.vertex_neighbor_vertices.html
    index_pointers = tri.vertex_neighbor_vertices[0]
    indices = tri.vertex_neighbor_vertices[1]

    # random coefficients must be negative numbers, size must be at least number of edges
    if constant_coefficients:
        random_values = -np.ones(shape=[tri.nsimplex * 3])
    else:
        if uniform:
            random_values = -np.random.uniform(size=[tri.nsimplex * 3])
        else:
            random_values = -np.exp(np.random.normal(size=[tri.nsimplex * 3]))
    A = lil_matrix((num_points, num_points), dtype=np.float64)

    # go through every point in the unit square, and add a random coefficient to it's inside neighbors.
    # if there is one or more outside neighbors, record the number so later we'll add this number of coefficient
    # to the diagonal
    boundary_indices = []
    for vertex_id in range(num_points):
        neighbors = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
        outside_neighbors = neighbors[np.where(neighbors >= num_points)]
        if len(outside_neighbors) > 0:
            boundary_indices.append(vertex_id)

    num_boundary_neighbors_dict = {}
    edge_counter = 0
    is_boundary = np.in1d(range(num_points), boundary_indices, assume_unique=True)
    for vertex_id in range(num_points):
        # if vertex is boundary, do not include in A
        # if np.isin(vertex_id, boundary_indices):
        #     continue
        if is_boundary[vertex_id]:
            continue

        neighbors = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
        internal_neighbors = np.setdiff1d(neighbors, boundary_indices, assume_unique=True)
        num_boundary_neighbors = len(neighbors) - len(internal_neighbors)
        if num_boundary_neighbors > 0:
            num_boundary_neighbors_dict[vertex_id] = num_boundary_neighbors

        for neighbor_id in np.sort(internal_neighbors):
            A[vertex_id, neighbor_id] = random_values[edge_counter]
            A[neighbor_id, vertex_id] = random_values[edge_counter]
            edge_counter += 1

    # set row sums to be zero
    row_sums = A.sum(axis=0)
    for i in range(num_points):
        A[i, i] = -row_sums[0, i]

    for boundary_id, num_boundary_neighbors in num_boundary_neighbors_dict.items():
        diagonal_value = -random_values[edge_counter:edge_counter + num_boundary_neighbors].sum()
        edge_counter += num_boundary_neighbors
        A[boundary_id, boundary_id] += diagonal_value

    # drop zero rows and columns
    if matlab_engine:
        A = drop_zero_row_col_matlab(A, matlab_engine)
    else:
        A = drop_zero_row_col(A).tocsr()
    return A


def generate_A_spec_cluster(num_unknowns, add_diag=False, num_clusters=2, unit_std=False, dim=2, dist='gauss', gamma=None,
                            distance=False, return_x=False, n_neighbors=10):
    """
    Similar params to https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
    With spectral clustering
    """
    centers = num_clusters
    if num_clusters == 2 and not unit_std:
        cluster_std = [1.0, 2.5]  # looks good, sometimes graph is connected, sometimes not
        size_factor = 1
    else:
        cluster_std = 1.0
        size_factor = num_unknowns / 1000
    center_box = [-10 * size_factor, 10 * size_factor]
    norm_laplacian = True

    if dist == 'gauss':
        X, y = datasets.make_blobs(n_samples=num_unknowns, n_features=dim, centers=centers,
                                   cluster_std=cluster_std, center_box=center_box)
    elif dist == 'moons':
        X, y = datasets.make_moons(n_samples=num_unknowns, noise=.05)
    elif dist == 'circles':
        X, y = datasets.make_circles(n_samples=num_unknowns, noise=.05, factor=.5)
    elif dist == 'random':
        X = np.random.rand(num_unknowns, dim)
    X = StandardScaler().fit_transform(X)

    if distance:
        mode = 'distance'
    else:
        mode = 'connectivity'
    connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, mode=mode,
                                    include_self=True)
    if gamma is not None:
        np.exp(-(gamma * connectivity.data) ** 2, out=connectivity.data)
    affinity_matrix = 0.5 * (connectivity + connectivity.T)

    laplacian, dd = csgraph_laplacian(affinity_matrix, normed=norm_laplacian,
                                      return_diag=True)
    # set diagonal to 1 if normed
    if norm_laplacian:
        diag_idx = (laplacian.row == laplacian.col)
        laplacian.data[diag_idx] = 1

    if add_diag:
        small_diag = scipy.sparse.diags(np.random.uniform(0, 0.02, num_unknowns))
        laplacian += small_diag

    if return_x:
        return X, laplacian
    else:
        return laplacian


def generate_A_delaunay_block_periodic_lognormal(num_unknowns_per_block, root_num_blocks, matlab_engine):
    """Poisson equation on triangular mesh, with lognormal coefficients, and block periodic boundary conditions"""
    # points are correct only for 3x3 number of blocks
    A_matlab, points_matlab = matlab_engine.block_periodic_delaunay(num_unknowns_per_block, root_num_blocks, nargout=2)
    A_numpy = np.array(A_matlab._data).reshape(A_matlab.size, order='F')
    points_numpy = np.array(points_matlab._data).reshape(points_matlab.size, order='F')
    return csr_matrix(A_numpy), points_numpy


def As_poisson_grid(num_As, num_unknowns, constant_coefficients=False):
    grid_size = int(math.sqrt(num_unknowns))
    if grid_size ** 2 != num_unknowns:
        raise RuntimeError("num_unknowns must be a square number")
    stencils = poisson_dirichlet_stencils(num_As, grid_size, constant_coefficients=constant_coefficients)
    A_idx, stencil_idx = compute_A_indices(grid_size)
    matrices = []
    for stencil in stencils:
        matrix = csr_matrix(arg1=(stencil.reshape((-1)), (A_idx[:, 0], A_idx[:, 1])),
                            shape=(grid_size ** 2, grid_size ** 2))
        matrix.eliminate_zeros()
        matrices.append(matrix)
    return matrices


@lru_cache(maxsize=None)
def create_hole_mesh(num_unknowns):
    def round_trip_connect(start, end):
        return [(i, i + 1) for i in range(start, end)] + [(end, start)]

    points = [(1 / 4, 0), (1 / 4, 1 / 4), (-1 / 4, 1 / 4), (-1 / 4, -1 / 4), (1 / 4, -1 / 4), (1 / 4, 0)]
    facets = round_trip_connect(0, len(points) - 1)

    circ_start = len(points)
    points.extend(
        (1.5 * np.cos(angle), 1.5 * np.sin(angle))
        for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False))

    facets.extend(round_trip_connect(circ_start, len(points) - 1))
    maximum_area = 0.5 / num_unknowns

    def needs_refinement(vertices, area):
        bary = np.sum(np.array(vertices), axis=0) / 3
        max_area = maximum_area + (np.linalg.norm(bary, np.inf) - 1 / 4) * maximum_area * 10
        return bool(area > max_area)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_holes([(0, 0)])
    info.set_facets(facets)

    mesh = triangle.build(info, refinement_func=needs_refinement)
    return mesh


@lru_cache(maxsize=None)
def create_quality_mesh(num_unknowns):
    mesh_info = triangle.MeshInfo()
    points = [(1.5, 1.5), (-1.5, 1.5), (-1.5, -1.5), (1.5, -1.5)]
    segments = [(i, i + 1) for i in range(3)] + [(3, 0)]
    mesh_info.set_points(points)
    mesh_info.set_facets(segments)
    mesh = triangle.build(mesh_info, max_volume=3 / (num_unknowns * 2 ** 2), min_angle=30)
    return mesh


def vertex_to_tris_map(simplices):
    M = defaultdict(set)
    for i, tri in enumerate(simplices):
        for point in tri:
            M[point].add(i)
    return M


def A_dirichlet_finite_element_quality(num_unknowns, matlab_engine, constant_coeffs=False, hole=False, uniform=False):
    if hole:
        mesh = create_hole_mesh(num_unknowns)
    else:
        mesh = create_quality_mesh(num_unknowns)
    mesh_points = np.array(mesh.points)
    num_points = mesh_points.shape[0]

    tris = np.array(mesh.elements)
    vertex_to_tris = vertex_to_tris_map(tris)

    if constant_coeffs:
        coeffs = np.ones(tris.shape[0])
    else:
        if uniform:
            coeffs = np.random.uniform(size=[tris.shape[0]])
        else:
            coeffs = np.exp(np.random.normal(scale=0.5, size=[tris.shape[0]]))
        # np.clip(coeffs, 0.1, 5, out=coeffs)
        # coeffs = np.random.uniform(size=tris.shape[0])

    x0 = mesh_points[tris[:, 0], 0]
    x1 = mesh_points[tris[:, 1], 0]
    x2 = mesh_points[tris[:, 2], 0]
    y0 = mesh_points[tris[:, 0], 1]
    y1 = mesh_points[tris[:, 1], 1]
    y2 = mesh_points[tris[:, 2], 1]

    a = x1 - x0
    b = y1 - y0
    c = x2 - x0
    d = y2 - y0

    s = np.abs((a * d - b * c) / 2)
    det_A = x0 * y1 - x0 * y2 - x1 * y0 + x1 * y2 + x2 * y0 - x2 * y1

    coeffs_per_tri = coeffs * s / det_A ** 2

    x = mesh_points[:, 0]
    y = mesh_points[:, 1]

    # lil_matrix is faster than dok_matrix in this case
    A = lil_matrix((num_points, num_points), dtype=np.float64)
    for p in range(num_points):
        point_tris = vertex_to_tris[p]
        for point_tri in point_tris:
            tri_points = tris[point_tri, :]
            tri_coeff = coeffs_per_tri[point_tri]
            ind_others = tri_points[tri_points != p]
            p1 = ind_others[0]
            p2 = ind_others[1]

            A[p, p] = A[p, p] + tri_coeff * ((y[p2] - y[p1]) ** 2 + (x[p2] - x[p1]) ** 2)
            A[p, p1] = A[p, p1] + tri_coeff * ((y[p1] - y[p2]) * (y[p2] - y[p]) + (x[p1] - x[p2]) * (x[p2] - x[p]))
            A[p, p2] = A[p, p2] + tri_coeff * ((y[p2] - y[p1]) * (y[p1] - y[p]) + (x[p2] - x[p1]) * (x[p1] - x[p]))

    if hole:
        outside_indices = np.where(
            (np.linalg.norm(mesh_points, axis=1) >= 1) |
            (np.linalg.norm(mesh_points, axis=1) <= 1 / 3)
        )[0]
    else:
        outside_indices = np.where(
            (mesh_points[:, 0] <= 0) |
            (mesh_points[:, 0] >= 1) |
            (mesh_points[:, 1] <= 0) |
            (mesh_points[:, 1] >= 1)
        )[0]
    A = drop_row_col_matlab(A, outside_indices, matlab_engine)
    return A.tocsr()


def poisson_dirichlet_stencils(num_stencils, grid_size, constant_coefficients=False):
    stencil = np.zeros((num_stencils, grid_size, grid_size, 3, 3))

    if constant_coefficients:
        diffusion_coeff = np.ones(shape=[num_stencils, grid_size, grid_size])
    else:
        diffusion_coeff = np.exp(np.random.normal(size=[num_stencils, grid_size, grid_size]))

    jm1 = [(i - 1) % grid_size for i in range(grid_size)]
    stencil[:, :, :, 1, 2] = -1. / 6 * (diffusion_coeff[:, jm1] + diffusion_coeff)
    stencil[:, :, :, 2, 1] = -1. / 6 * (diffusion_coeff + diffusion_coeff[:, :, jm1])
    stencil[:, :, :, 2, 0] = -1. / 3 * diffusion_coeff[:, :, jm1]
    stencil[:, :, :, 2, 2] = -1. / 3 * diffusion_coeff

    jp1 = [(i + 1) % grid_size for i in range(grid_size)]

    stencil[:, :, :, 1, 0] = stencil[:, :, jm1, 1, 2]
    stencil[:, :, :, 0, 0] = stencil[:, jm1][:, :, jm1][:, :, :, 2, 2]
    stencil[:, :, :, 0, 1] = stencil[:, jm1][:, :, :, 2, 1]
    stencil[:, :, :, 0, 2] = stencil[:, jm1][:, :, jp1][:, :, :, 2, 0]
    stencil[:, :, :, 1, 1] = -np.sum(np.sum(stencil, axis=4), axis=3)

    stencil[:, :, 0, :, 0] = 0.
    stencil[:, :, -1, :, -1] = 0.
    stencil[:, 0, :, 0, :] = 0.
    stencil[:, -1, :, -1, :] = 0.
    return stencil


@lru_cache(maxsize=None)
def compute_A_indices(grid_size):
    K = map_2_to_1(grid_size=grid_size)
    A_idx = []
    stencil_idx = []
    for i in range(grid_size):
        for j in range(grid_size):
            I = int(K[i, j, 1, 1])
            for k in range(3):
                for m in range(3):
                    J = int(K[i, j, k, m])
                    A_idx.append([I, J])
                    stencil_idx.append([i, j, k, m])
    return np.array(A_idx), stencil_idx


def map_2_to_1(grid_size):
    # maps 2D coordinates to the corresponding 1D coordinate in the matrix.
    k = np.zeros((grid_size, grid_size, 3, 3))
    M = np.reshape(np.arange(grid_size ** 2), (grid_size, grid_size)).T
    M = np.concatenate([M, M], 0)
    M = np.concatenate([M, M], 1)
    for i in range(3):
        I = (i - 1) % grid_size
        for j in range(3):
            J = (j - 1) % grid_size
            k[:, :, i, j] = M[I:I + grid_size, J:J + grid_size]
    return k


def create_mesh(num_nodes):
    # create uniformly random points on the unit square
    rand_points = np.random.uniform([0, 0], [1, 1], [num_nodes, 2])
    return Delaunay(rand_points)


def generate_A_delaunay_periodic_lognormal(num_unknowns, uniform=False):
    tri = create_mesh(num_unknowns)
    verts = tri.vertices
    if uniform:
        random_values = -np.random.uniform(size=[tri.nsimplex, 3])
    else:
        random_values = -np.exp(np.random.normal(size=[tri.nsimplex, 3]))  # must be negative numbers
    A = lil_matrix((num_unknowns, num_unknowns), dtype=np.float64)

    for i in range(tri.nsimplex):
        vert = verts[i]
        A[vert[0], vert[1]] = random_values[i, 0]
        A[vert[0], vert[2]] = random_values[i, 1]
        A[vert[1], vert[2]] = random_values[i, 2]

        # symmetrize
        A[vert[1], vert[0]] = random_values[i, 0]
        A[vert[2], vert[0]] = random_values[i, 1]
        A[vert[2], vert[1]] = random_values[i, 2]

    # set row sums to be zero
    row_sums = A.sum(axis=0)
    for i in range(num_unknowns):
        A[i, i] = -row_sums[0, i]
    return A.tocsr()


def drop_row_col_matlab(A, indices, matlab_engine):
    size = A.shape[0]
    A_coo = A.tocoo()
    A_rows = matlab.double((A_coo.row + 1))
    A_cols = matlab.double((A_coo.col + 1))
    indices = matlab.double((indices + 1))
    A_values = matlab.double(A_coo.data)
    rows, cols, values = matlab_engine.drop_row_col(A_rows, A_cols, A_values, size, indices, nargout=3)
    rows = np.array(rows._data).reshape(rows.size, order='F') - 1
    cols = np.array(cols._data).reshape(cols.size, order='F') - 1
    values = np.array(values._data).reshape(values.size, order='F')
    rows, cols, values = rows.T[0], cols.T[0], values.T[0]
    rows, cols = rows.astype(np.int), cols.astype(np.int)
    return csr_matrix((values, (rows, cols)))
