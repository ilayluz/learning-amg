import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix


def to_sparse(a):
    # tf.contrib.layers.dense_to_sparse has a memory leak issue, see:
    # https://github.com/tensorflow/tensorflow/issues/17590
    zero = tf.constant(0, a.dtype)
    indices = tf.where(tf.not_equal(a, zero))
    values = tf.gather_nd(a, indices)
    shape = a.shape
    return tf.SparseTensor(indices, values, shape)


def to_dense(a):
    # the tf.sparse.to_dense function does not support back-propagation, see:
    # https://github.com/tensorflow/tensorflow/issues/6391
    # bug is fixed in Tensorflow 1.14: https://github.com/tensorflow/tensorflow/releases/tag/v1.14.0
    # for older versions, workaround is performing addition with dense zero tensor
    # return tf.sparse.add(a, tf.zeros(a.shape, dtype=a.dtype))
    return tf.sparse.to_dense(a)


def csr_to_sparse_tensor(a):
    # https://stackoverflow.com/a/43130239
    a_coo = a.tocoo()
    a_coo.eliminate_zeros()
    indices = np.mat([a_coo.row, a_coo.col]).transpose()
    tensor = tf.SparseTensor(indices, a_coo.data, a_coo.shape)
    return tf.sparse.reorder(tensor)
    # return tensor


def sparse_tensor_to_csr(a):
    indices = a.indices.numpy()
    rows = indices[:, 0]
    cols = indices[:, 1]
    data = a.values.numpy()
    shape = (a.shape[0].value, a.shape[1].value)
    a_coo = coo_matrix((data, (rows, cols)), shape=shape)
    return a_coo.tocsr()


def sparse_multiply(a, b):
    #  there are multiple ways to do it in Tensorflow, see:
    #  https://www.tensorflow.org/api_docs/python/tf/sparse/sparse_dense_matmul
    #  https://stackoverflow.com/questions/34030140/is-sparse-tensor-multiplication-implemented-in-tensorflow
    dense_a = to_dense(a)
    dense_b = to_dense(b)
    return tf.matmul(dense_a, dense_b, a_is_sparse=True, b_is_sparse=True)


def pad_diagonal(a, padded_length):
    # given square matrix "a", pad with 1's on diagonal until matrix is of size "padded_length"
    a_length = a.shape[0].value
    if a_length > padded_length:
        raise RuntimeError(f"padded length {padded_length} is larger than matrix length {a_length}")
    if a_length == padded_length:
        return a

    a_indices = a.indices
    new_range = tf.range(a_length, padded_length, dtype=tf.int64)
    new_indices = tf.tile(tf.expand_dims(new_range, axis=1), [1, 2])
    padded_indices = tf.concat([a_indices, new_indices], axis=0)

    a_values = a.values
    new_values = tf.ones(padded_length - a_length, dtype=tf.float64)
    padded_values = tf.concat([a_values, new_values], axis=0)

    padded_shape = [padded_length, padded_length]
    return tf.SparseTensor(indices=padded_indices, values=padded_values, dense_shape=padded_shape)

