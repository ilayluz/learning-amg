import tensorflow as tf
from scipy.sparse import csr_matrix

from model import csrs_to_graphs_tuple, graphs_tuple_to_sparse_tensor, to_prolongation_matrix_csr
from tf_sparse_utils import sparse_tensor_to_csr


def graphs_tuple_to_csr(graphs_tuple):
    row_indices = graphs_tuple.senders.numpy()
    col_indices = graphs_tuple.receivers.numpy()
    data = tf.squeeze(graphs_tuple.edges).numpy()
    num_nodes = graphs_tuple.n_node.numpy()[0]
    shape = (num_nodes, num_nodes)
    return csr_matrix((data, (row_indices, col_indices)), shape=shape)


def model(A, coarse_nodes, baseline_P, C, graph_model, matlab_engine=None, normalize_rows_by_node=False,
          edge_indicators=True, node_indicators=True):
    with tf.device(":/gpu:0"):
        graphs_tuple = csrs_to_graphs_tuple([A], matlab_engine, coarse_nodes_list=[coarse_nodes],
                                            baseline_P_list=[baseline_P],
                                            edge_indicators=edge_indicators,
                                            node_indicators=node_indicators)
        output_graph = graph_model(graphs_tuple)
    P_square_tensor = graphs_tuple_to_sparse_tensor(output_graph)
    nodes_tensor = tf.squeeze(output_graph.nodes)
    nodes = nodes_tensor.numpy()

    P_square_csr = sparse_tensor_to_csr(P_square_tensor)
    P_csr = to_prolongation_matrix_csr(P_square_csr, coarse_nodes, baseline_P, nodes,
                                       normalize_rows_by_node=normalize_rows_by_node)
    return P_csr


def baseline(A, splitting, baseline_P, C):
    return baseline_P

