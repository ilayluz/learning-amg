import graph_nets as gn
import matlab
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

from data import As_poisson_grid
from graph_net_model import EncodeProcessDecodeNonRecurrent


def get_model(model_name, model_config, run_config, matlab_engine, train=False, train_config=None):
    dummy_input = As_poisson_grid(1, 7 ** 2)[0]
    checkpoint_dir = './training_dir/' + model_name
    graph_model, optimizer, global_step = load_model(checkpoint_dir, dummy_input, model_config,
                                                     run_config,
                                                     matlab_engine, get_optimizer=train,
                                                     train_config=train_config)
    if train:
        return graph_model, optimizer, global_step
    else:
        return graph_model


def load_model(checkpoint_dir, dummy_input, model_config, run_config, matlab_engine, get_optimizer=True,
               train_config=None):
    tf.enable_eager_execution()
    model = create_model(model_config)

    # we have to use the model at least once to get the list of variables
    model(csrs_to_graphs_tuple([dummy_input], matlab_engine, coarse_nodes_list=np.array([[0, 1]]),
                               baseline_P_list=[tf.convert_to_tensor(dummy_input.toarray()[:, [0, 1]])],
                               node_indicators=run_config.node_indicators,
                               edge_indicators=run_config.edge_indicators))

    variables = model.get_all_variables()
    variables_dict = {variable.name: variable for variable in variables}
    if get_optimizer:
        global_step = tf.train.get_or_create_global_step()
        decay_steps = 100
        decay_rate = 1.0
        learning_rate = tf.train.exponential_decay(train_config.learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        checkpoint = tf.train.Checkpoint(**variables_dict, optimizer=optimizer, global_step=global_step)
    else:
        optimizer = None
        global_step = None
        checkpoint = tf.train.Checkpoint(**variables_dict)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is None:
        raise RuntimeError(f'training_dir {checkpoint_dir} does not exist')
    checkpoint.restore(latest_checkpoint)
    return model, optimizer, global_step


def create_model(model_config):
    with tf.device('/gpu:0'):
        return EncodeProcessDecodeNonRecurrent(num_cores=model_config.mp_rounds, edge_output_size=1,
                                               node_output_size=1, global_block=model_config.global_block,
                                               latent_size=model_config.latent_size,
                                               num_layers=model_config.mlp_layers,
                                               concat_encoder=model_config.concat_encoder)


def csrs_to_graphs_tuple(csrs, matlab_engine, node_feature_size=128, coarse_nodes_list=None, baseline_P_list=None,
                         node_indicators=True, edge_indicators=True):
    dtype = tf.float64

    # build up the arguments for the GraphsTuple constructor
    n_node = tf.convert_to_tensor([csr.shape[0] for csr in csrs])
    n_edge = tf.convert_to_tensor([csr.nnz for csr in csrs])

    if not edge_indicators:
        numpy_edges = np.concatenate([csr.data for csr in csrs])
        edges = tf.expand_dims(tf.convert_to_tensor(numpy_edges, dtype=dtype), axis=1)
    else:
        edge_encodings_list = []
        for csr, coarse_nodes, baseline_P in zip(csrs, coarse_nodes_list, baseline_P_list):
            if tf.is_tensor(baseline_P):
                baseline_P = csr_matrix(baseline_P.numpy())

            baseline_P_rows, baseline_P_cols = P_square_sparsity_pattern(baseline_P, baseline_P.shape[0],
                                                                         coarse_nodes, matlab_engine)
            coo = csr.tocoo()

            # construct numpy structured arrays, where each element is a tuple (row,col), so that we can later use
            # the numpy set function in1d()
            baseline_P_indices = np.core.records.fromarrays([baseline_P_rows, baseline_P_cols], dtype='i,i')
            coo_indices = np.core.records.fromarrays([coo.row, coo.col], dtype='i,i')

            same_indices = np.in1d(coo_indices, baseline_P_indices, assume_unique=True)
            baseline_edges = same_indices.astype(np.float64)
            non_baseline_edges = (~same_indices).astype(np.float64)

            edge_encodings = np.stack([coo.data, baseline_edges, non_baseline_edges]).T
            edge_encodings_list.append(edge_encodings)
        numpy_edges = np.concatenate(edge_encodings_list)
        edges = tf.convert_to_tensor(numpy_edges, dtype=dtype)

    # COO format for sparse matrices contains a list of row indices and a list of column indices
    coos = [csr.tocoo() for csr in csrs]
    senders_numpy = np.concatenate([coo.row for coo in coos])
    senders = tf.convert_to_tensor(senders_numpy)
    receivers_numpy = np.concatenate([coo.col for coo in coos])
    receivers = tf.convert_to_tensor(receivers_numpy)

    # see the source of _concatenate_data_dicts for explanation
    offsets = gn.utils_tf._compute_stacked_offsets(n_node, n_edge)
    senders += offsets
    receivers += offsets

    if not node_indicators:
        nodes = None
    else:
        node_encodings_list = []
        for csr, coarse_nodes in zip(csrs, coarse_nodes_list):
            coarse_indices = np.in1d(range(csr.shape[0]), coarse_nodes, assume_unique=True)

            coarse_node_encodings = coarse_indices.astype(np.float64)
            fine_node_encodings = (~coarse_indices).astype(np.float64)
            node_encodings = np.stack([coarse_node_encodings, fine_node_encodings]).T

            node_encodings_list.append(node_encodings)

        numpy_nodes = np.concatenate(node_encodings_list)
        nodes = tf.convert_to_tensor(numpy_nodes, dtype=dtype)

    graphs_tuple = gn.graphs.GraphsTuple(
        nodes=nodes,
        edges=edges,
        globals=None,
        receivers=receivers,
        senders=senders,
        n_node=n_node,
        n_edge=n_edge
    )
    if not node_indicators:
        graphs_tuple = gn.utils_tf.set_zero_node_features(graphs_tuple, 1, dtype=dtype)

    graphs_tuple = gn.utils_tf.set_zero_global_features(graphs_tuple, node_feature_size, dtype=dtype)

    return graphs_tuple


def P_square_sparsity_pattern(P, size, coarse_nodes, matlab_engine):
    P_coo = P.tocoo()
    P_rows = matlab.double((P_coo.row + 1))
    P_cols = matlab.double((P_coo.col + 1))
    P_values = matlab.double(P_coo.data)
    coarse_nodes = matlab.double((coarse_nodes + 1))
    rows, cols = matlab_engine.square_P(P_rows, P_cols, P_values, size, coarse_nodes,  nargout=2)
    rows = np.array(rows._data).reshape(rows.size, order='F') - 1
    cols = np.array(cols._data).reshape(cols.size, order='F') - 1
    rows, cols = rows.T[0], cols.T[0]
    return rows, cols


def graphs_tuple_to_sparse_tensor(graphs_tuple):
    senders = graphs_tuple.senders
    receivers = graphs_tuple.receivers
    indices = tf.cast(tf.stack([senders, receivers], axis=1), tf.int64)

    # first element in the edge feature is the value, the other elements are metadata
    values = tf.squeeze(graphs_tuple.edges[:, 0])

    shape = tf.concat([graphs_tuple.n_node, graphs_tuple.n_node], axis=0)
    shape = tf.cast(shape, tf.int64)

    matrix = tf.sparse.SparseTensor(indices, values, shape)
    # reordering is required because the pyAMG coarsening step does not preserve indices order
    matrix = tf.sparse.reorder(matrix)

    return matrix


def to_prolongation_matrix_csr(matrix, coarse_nodes, baseline_P, nodes, normalize_rows=True,
                               normalize_rows_by_node=False):
    """
    sparse version of the above function, for when the dense matrix is too large to fit in GPU memory
    used only for inference, so no need for backpropagation, inputs are csr matrices
    """
    # prolongation from coarse point to itself should be identity. This corresponds to 1's on the diagonal
    matrix.setdiag(np.ones(matrix.shape[0]))

    # select only columns corresponding to coarse nodes
    matrix = matrix[:, coarse_nodes]

    # set sparsity pattern (interpolatory sets) to be of baseline prolongation
    baseline_P_mask = (baseline_P != 0).astype(np.float64)
    matrix = matrix.multiply(baseline_P_mask)
    matrix.eliminate_zeros()

    if normalize_rows:
        if normalize_rows_by_node:
            baseline_row_sum = nodes
        else:
            baseline_row_sum = baseline_P.sum(axis=1)
            baseline_row_sum = np.array(baseline_row_sum)[:, 0]

        matrix_row_sum = np.array(matrix.sum(axis=1))[:, 0]
        # https://stackoverflow.com/a/12238133
        matrix_copy = matrix.copy()
        matrix_copy.data /= matrix_row_sum.repeat(np.diff(matrix_copy.indptr))
        matrix_copy.data *= baseline_row_sum.repeat(np.diff(matrix_copy.indptr))
        matrix = matrix_copy
    return matrix


def to_prolongation_matrix_tensor(matrix, coarse_nodes, baseline_P, nodes,
                                  normalize_rows=True,
                                  normalize_rows_by_node=False):
    dtype = tf.float64
    matrix = tf.cast(matrix, dtype)
    matrix = tf.sparse.to_dense(matrix)

    # prolongation from coarse point to itself should be identity. This corresponds to 1's on the diagonal
    matrix = tf.linalg.set_diag(matrix, tf.ones(matrix.shape[0], dtype=dtype))

    # select only columns corresponding to coarse nodes
    matrix = tf.gather(matrix, coarse_nodes, axis=1)

    # set sparsity pattern (interpolatory sets) to be of baseline prolongation
    baseline_zero_mask = tf.cast(tf.not_equal(baseline_P, tf.zeros_like(baseline_P)), dtype)
    matrix = matrix * baseline_zero_mask

    if normalize_rows:
        if normalize_rows_by_node:
            baseline_row_sum = nodes
        else:
            baseline_row_sum = tf.reduce_sum(baseline_P, axis=1)
        baseline_row_sum = tf.cast(baseline_row_sum, dtype)

        matrix_row_sum = tf.reduce_sum(matrix, axis=1)
        matrix_row_sum = tf.cast(matrix_row_sum, dtype)

        # there might be a few rows that are all 0's - corresponding to fine points that are not connected to any
        # coarse point. We use "divide_no_nan" to let these rows remain 0's
        matrix = tf.math.divide_no_nan(matrix, tf.reshape(matrix_row_sum, (-1, 1)))

        matrix = matrix * tf.reshape(baseline_row_sum, (-1, 1))
    return matrix


def graphs_tuple_to_sparse_matrices(graphs_tuple, return_nodes=False):
    num_graphs = int(graphs_tuple.n_node.shape[0])
    graphs = [gn.utils_tf.get_graph(graphs_tuple, i)
              for i in range(num_graphs)]

    matrices = [graphs_tuple_to_sparse_tensor(graph) for graph in graphs]

    if return_nodes:
        nodes_list = [tf.squeeze(graph.nodes) for graph in graphs]
        return matrices, nodes_list
    else:
        return matrices


