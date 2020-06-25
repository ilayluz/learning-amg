import copy
import os
import random
import string

import fire
import matlab.engine
import numpy as np
import pyamg
import tensorflow as tf
from pyamg.classical import direct_interpolation
from scipy.sparse import csr_matrix
from tqdm import tqdm

import configs
from data import generate_A
from dataset import DataSet
from model import csrs_to_graphs_tuple, create_model, graphs_tuple_to_sparse_matrices, to_prolongation_matrix_tensor
from multigrid_utils import block_diagonalize_A_single, block_diagonalize_P, two_grid_error_matrices, frob_norm, \
    two_grid_error_matrix, compute_coarse_A
from relaxation import relaxation_matrices
from utils import create_results_dir, write_config_file, most_frequent_splitting, chunks


def create_dataset(num_As, data_config, run=0, matlab_engine=None):
    if data_config.load_data:
        As_filename = f"data_dir/periodic_delaunay_num_As_{num_As}_num_points_{data_config.num_unknowns}" \
            f"_rnb_{data_config.root_num_blocks}_epoch_{run}.npy"
        if not os.path.isfile(As_filename):
            raise RuntimeError(f"file {As_filename} not found")
        As = np.load(As_filename)

        # workaround for data generated with both matrices and point coordinates
        if len(As.shape) == 1:
            As = list(As)
        elif len(As.shape) == 2:
            As = list(As[0])
    else:
        As = [generate_A(data_config.num_unknowns,
                         data_config.dist,
                         data_config.block_periodic,
                         data_config.root_num_blocks,
                         add_diag=data_config.add_diag,
                         matlab_engine=matlab_engine) for _ in range(num_As)]

    if data_config.save_data:
        As_filename = f"data_dir/periodic_delaunay_num_As_{num_As}_num_points_{data_config.num_unknowns}" \
            f"_rnb_{data_config.root_num_blocks}_epoch_{run}.npy"
        np.save(As_filename, As)
    return create_dataset_from_As(As, data_config)


def create_dataset_from_As(As, data_config):
    if data_config.block_periodic:
        Ss = [None] * len(As)  # relaxation matrices are only created per block when calling loss()
    else:
        Ss = relaxation_matrices(As)
    if data_config.block_periodic:
        orig_solvers = [pyamg.ruge_stuben_solver(A, max_levels=2, keep=True, CF=data_config.splitting)
                        for A in As]
        # for efficient Fourier analysis, we require that each block contains the same sparsity pattern - set of
        # coarse nodes, and interpolatory set for each node. The AMG C/F splitting algorithms do not output the same
        # splitting for each block, but the blocks are relatively similar to each other. Taking the most common set
        # of coarse nodes and repeating it for each block might be a good strategy
        splittings = []
        baseline_P_list = []
        for i in range(len(As)):
            # visualize_cf_splitting(As[i], Vs[i], orig_splittings[i])

            orig_splitting = orig_solvers[i].levels[0].splitting
            block_splittings = list(chunks(orig_splitting, data_config.num_unknowns))
            common_block_splitting = most_frequent_splitting(block_splittings)
            repeated_splitting = np.tile(common_block_splitting, data_config.root_num_blocks ** 2)
            splittings.append(repeated_splitting)

            # we recompute the Ruge-Stuben prolongation matrix with the modified splitting, and the original strength
            # matrix. We assume the strength matrix is block-circulant (because A is block-circulant)
            A = As[i]
            C = orig_solvers[i].levels[0].C
            P = direct_interpolation(A, C, repeated_splitting)
            baseline_P_list.append(tf.convert_to_tensor(P.toarray(), dtype=tf.float64))

        coarse_nodes_list = [np.nonzero(splitting)[0] for splitting in splittings]

    else:
        solvers = [pyamg.ruge_stuben_solver(A, max_levels=2, keep=True, CF=data_config.splitting)
                   for A in As]
        baseline_P_list = [solver.levels[0].P for solver in solvers]
        baseline_P_list = [tf.convert_to_tensor(P.toarray(), dtype=tf.float64) for P in baseline_P_list]
        splittings = [solver.levels[0].splitting for solver in solvers]
        coarse_nodes_list = [np.nonzero(splitting)[0] for splitting in splittings]
    return DataSet(As, Ss, coarse_nodes_list, baseline_P_list)


def loss(dataset, A_graphs_tuple, P_graphs_tuple,
         run_config, train_config, data_config):
    As = graphs_tuple_to_sparse_matrices(A_graphs_tuple)
    Ps_square, nodes_list = graphs_tuple_to_sparse_matrices(P_graphs_tuple, True)

    if train_config.fourier:
        As = [tf.cast(tf.sparse.to_dense(A), tf.complex128) for A in As]
        block_As = [block_diagonalize_A_single(A, data_config.root_num_blocks, tensor=True) for A in As]
        block_Ss = relaxation_matrices([csr_matrix(A.numpy()) for block_A in block_As for A in block_A])

    batch_size = len(dataset.coarse_nodes_list)
    total_norm = tf.Variable(0.0, dtype=tf.float64)
    for i in range(batch_size):
        if train_config.fourier:
            num_blocks = data_config.root_num_blocks ** 2 - 1

            P_square = Ps_square[i]
            coarse_nodes = dataset.coarse_nodes_list[i]
            baseline_P = dataset.baseline_P_list[i]
            nodes = nodes_list[i]
            P = to_prolongation_matrix_tensor(P_square, coarse_nodes, baseline_P, nodes,
                                              normalize_rows=run_config.normalize_rows,
                                              normalize_rows_by_node=run_config.normalize_rows_by_node)
            block_P = block_diagonalize_P(P, data_config.root_num_blocks, coarse_nodes)

            As = tf.stack(block_As[i])
            Ps = tf.stack(block_P)
            Rs = tf.transpose(Ps, perm=[0, 2, 1], conjugate=True)
            Ss = tf.convert_to_tensor(block_Ss[num_blocks * i:num_blocks * (i + 1)])

            Ms = two_grid_error_matrices(As, Ps, Rs, Ss)
            M = Ms[-1]  # for logging
            block_norms = tf.abs(frob_norm(Ms, power=1))

            block_max_norm = tf.reduce_max(block_norms)
            total_norm = total_norm + block_max_norm

        else:
            A = tf.sparse.to_dense(As[i])
            P_square = Ps_square[i]
            coarse_nodes = dataset.coarse_nodes_list[i]
            baseline_P = dataset.baseline_P_list[i]
            nodes = nodes_list[i]
            P = to_prolongation_matrix_tensor(P_square, coarse_nodes, baseline_P, nodes,
                                              normalize_rows=run_config.normalize_rows,
                                              normalize_rows_by_node=run_config.normalize_rows_by_node)
            R = tf.transpose(P)
            S = tf.convert_to_tensor(dataset.Ss[i])

            M = two_grid_error_matrix(A, P, R, S)

            norm = frob_norm(M, power=1)
            total_norm = total_norm + norm

    return total_norm / batch_size, M  # M is chosen randomly - the last in the batch


def save_model_and_optimizer(checkpoint_prefix, model, optimizer, global_step):
    variables = model.get_all_variables()
    variables_dict = {variable.name: variable for variable in variables}
    checkpoint = tf.train.Checkpoint(**variables_dict, optimizer=optimizer, global_step=global_step)
    checkpoint.save(file_prefix=checkpoint_prefix)
    return checkpoint


def train_run(run_dataset, run, batch_size, config,
              model, optimizer, global_step, checkpoint_prefix,
              eval_dataset, eval_A_graphs_tuple, eval_config, matlab_engine):
    num_As = len(run_dataset.As)
    if num_As % batch_size != 0:
        raise RuntimeError("batch size must divide training data size")

    run_dataset = run_dataset.shuffle()
    num_batches = num_As // batch_size
    loop = tqdm(range(num_batches))
    for batch in loop:
        start_index = batch * batch_size
        end_index = start_index + batch_size
        batch_dataset = run_dataset[start_index:end_index]

        batch_A_graphs_tuple = csrs_to_graphs_tuple(batch_dataset.As, matlab_engine,
                                                    coarse_nodes_list=batch_dataset.coarse_nodes_list,
                                                    baseline_P_list=batch_dataset.baseline_P_list,
                                                    node_indicators=config.run_config.node_indicators,
                                                    edge_indicators=config.run_config.edge_indicators)

        with tf.GradientTape() as tape:
            with tf.device('/gpu:0'):
                batch_P_graphs_tuple = model(batch_A_graphs_tuple)
            frob_loss, M = loss(batch_dataset, batch_A_graphs_tuple, batch_P_graphs_tuple,
                                config.run_config, config.train_config, config.data_config)

        print(f"frob loss: {frob_loss.numpy()}")
        save_every = max(1000 // batch_size, 1)
        if batch % save_every == 0:
            checkpoint = save_model_and_optimizer(checkpoint_prefix, model, optimizer, global_step)

        # we don't call .get_variables() because the model is Sequential/custom,
        # see docs for Sequential.get_variables()
        variables = model.get_all_variables()
        grads = tape.gradient(frob_loss, variables)

        global_step.assign_add(batch_size - 1)  # apply_gradients increments global_step by 1
        optimizer.apply_gradients(zip(grads, variables),
                                  global_step=global_step)

        record_tb(M, run, num_As, batch, batch_size, frob_loss, grads, loop, model,
                  variables, eval_dataset, eval_A_graphs_tuple, eval_config)
    return checkpoint


def record_tb_loss(frob_loss):
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('loss', frob_loss)


def record_tb_params(batch_size, grads, loop, variables):
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        if loop.avg_time is not None:
            tf.contrib.summary.scalar('seconds_per_batch', tf.convert_to_tensor(loop.avg_time))

        for i in range(len(variables)):
            variable = variables[i]
            variable_name = variable.name
            grad = grads[i]
            if grad is not None:
                tf.contrib.summary.scalar(variable_name + '_grad', tf.norm(grad) / batch_size)
                tf.contrib.summary.histogram(variable_name + '_grad_histogram', grad / batch_size)
                tf.contrib.summary.scalar(variable_name + '_grad_fraction_dead', tf.nn.zero_fraction(grad))
                tf.contrib.summary.scalar(variable_name + '_value', tf.norm(variable))
                tf.contrib.summary.histogram(variable_name + '_value_histogram', variable)


def record_tb_spectral_radius(M, model, eval_dataset, eval_A_graphs_tuple, eval_config):
    with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        spectral_radius = np.abs(np.linalg.eigvals(M.numpy())).max()
        tf.contrib.summary.scalar('spectral_radius', spectral_radius)

        with tf.device('/gpu:0'):
            eval_P_graphs_tuple = model(eval_A_graphs_tuple)
        eval_loss, eval_M = loss(eval_dataset, eval_A_graphs_tuple, eval_P_graphs_tuple,
                                 eval_config.run_config,
                                 eval_config.train_config,
                                 eval_config.data_config)

        eval_spectral_radius = np.abs(np.linalg.eigvals(eval_M.numpy())).max()
        tf.contrib.summary.scalar('eval_loss', eval_loss)
        tf.contrib.summary.scalar('eval_spectral_radius', eval_spectral_radius)


def record_tb(M, run, num_As, batch, batch_size, frob_loss, grads, loop, model,
              variables, eval_dataset, eval_A_graphs_tuple, eval_config):
    batch = run * num_As + batch

    record_loss_every = max(1 // batch_size, 1)
    if batch % record_loss_every == 0:
        record_tb_loss(frob_loss)

    record_params_every = max(300 // batch_size, 1)
    if batch % record_params_every == 0:
        record_tb_params(batch_size, grads, loop, variables)

    record_spectral_every = max(300 // batch_size, 1)
    if batch % record_spectral_every == 0:
        record_tb_spectral_radius(M, model, eval_dataset, eval_A_graphs_tuple, eval_config)


def clone_model(model, model_config, run_config, matlab_engine):
    clone = create_model(model_config)

    dummy_A = pyamg.gallery.poisson((7, 7), type='FE', format='csr')
    dummy_input = csrs_to_graphs_tuple([dummy_A], matlab_engine, coarse_nodes_list=np.array([[0, 1]]),
                                       baseline_P_list=[tf.convert_to_tensor(dummy_A.toarray()[:, [0, 1]])],
                                       node_indicators=run_config.node_indicators,
                                       edge_indicators=run_config.edge_indicators)
    clone(dummy_input)
    [var_clone.assign(var_orig) for var_clone, var_orig in zip(clone.get_all_variables(), model.get_all_variables())]
    return clone


def coarsen_As(fine_dataset, model, run_config, matlab_engine, batch_size=64):
    # computes the Galerkin operator P^(T)AP on each of the A matrices in a batch, using the Prolongation
    # outputted from the model
    As = fine_dataset.As
    coarse_nodes_list = fine_dataset.coarse_nodes_list
    baseline_P_list = fine_dataset.baseline_P_list

    batch_size = min(batch_size, len(As))
    num_batches = len(As) // batch_size

    batched_As = list(chunks(As, batch_size))
    batched_coarse_nodes_list = list(chunks(coarse_nodes_list, batch_size))
    batched_baseline_P_list = list(chunks(baseline_P_list, batch_size))
    A_graphs_tuple_batches = [csrs_to_graphs_tuple(batch_As, matlab_engine, coarse_nodes_list=batch_coarse_nodes_list,
                                                   baseline_P_list=batch_baseline_P_list,
                                                   node_indicators=run_config.node_indicators,
                                                   edge_indicators=run_config.edge_indicators
                                                   )
                              for batch_As, batch_coarse_nodes_list, batch_baseline_P_list
                              in zip(batched_As, batched_coarse_nodes_list, batched_baseline_P_list)]

    Ps_square = []
    nodes_list = []
    for batch in tqdm(range(num_batches)):
        A_graphs_tuple = A_graphs_tuple_batches[batch]
        P_graphs_tuple = model(A_graphs_tuple)
        P_square_batch, nodes_batch = graphs_tuple_to_sparse_matrices(P_graphs_tuple, return_nodes=True)
        Ps_square.extend(P_square_batch)
        nodes_list.extend(nodes_batch)

    coarse_As = []
    for i in tqdm(range(len(As))):
        P_square = Ps_square[i]
        nodes = nodes_list[i]
        coarse_nodes = coarse_nodes_list[i]
        baseline_P = baseline_P_list[i]
        P = to_prolongation_matrix_tensor(P_square, coarse_nodes, baseline_P, nodes)
        R = tf.transpose(P)
        A_csr = As[i]
        A = tf.convert_to_tensor(A_csr.toarray(), dtype=tf.float64)
        tensor_coarse_A = compute_coarse_A(R, A, P)
        coarse_A = csr_matrix(tensor_coarse_A.numpy())
        coarse_As.append(coarse_A)
    return coarse_As


def create_coarse_dataset(fine_dataset, model, data_config, run_config, matlab_engine):
    As = coarsen_As(fine_dataset, model, run_config, matlab_engine)
    return create_dataset_from_As(As, data_config)


def train(config='GRAPH_LAPLACIAN_TRAIN', eval_config='GRAPH_LAPLACIAN_EVAL', seed=1):
    config = getattr(configs, config)
    eval_config = getattr(configs, eval_config)
    eval_config.run_config = config.run_config

    matlab_engine = matlab.engine.start_matlab()

    # fix random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    matlab_engine.eval(f'rng({seed})')

    batch_size = min(config.train_config.samples_per_run, config.train_config.batch_size)

    # we measure the performance of the model over time on one larger instance that is not optimized for
    eval_dataset = create_dataset(1, eval_config.data_config)
    eval_A_graphs_tuple = csrs_to_graphs_tuple(eval_dataset.As, matlab_engine,
                                               coarse_nodes_list=eval_dataset.coarse_nodes_list,
                                               baseline_P_list=eval_dataset.baseline_P_list,
                                               node_indicators=eval_config.run_config.node_indicators,
                                               edge_indicators=eval_config.run_config.edge_indicators
                                               )

    if config.train_config.load_model:
        raise NotImplementedError()
    else:
        model = create_model(config.model_config)
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=config.train_config.learning_rate)

    run_name = ''.join(random.choices(string.digits, k=5))  # to make the run_name string unique
    create_results_dir(run_name)
    write_config_file(run_name, config, seed)

    checkpoint_prefix = os.path.join(config.train_config.checkpoint_dir + '/' + run_name, 'ckpt')
    log_dir = config.train_config.tensorboard_dir + '/' + run_name
    writer = tf.contrib.summary.create_file_writer(log_dir)
    writer.set_as_default()

    for run in range(config.train_config.num_runs):
        # we create the data before the training loop starts for efficiency,
        # at the loop we only slice batches and convert to tensors
        run_dataset = create_dataset(config.train_config.samples_per_run, config.data_config,
                                     run=run, matlab_engine=matlab_engine)

        checkpoint = train_run(run_dataset, run, batch_size, config,
                               model, optimizer, global_step,
                               checkpoint_prefix,
                               eval_dataset, eval_A_graphs_tuple, eval_config,
                               matlab_engine)
        checkpoint.save(file_prefix=checkpoint_prefix)

    if config.train_config.coarsen:
        old_model = clone_model(model, config.model_config, config.run_config, matlab_engine)

        for run in range(config.train_config.num_runs):
            run_dataset = create_dataset(config.train_config.samples_per_run, config.data_config,
                                         run=run, matlab_engine=matlab_engine)

            fine_data_config = copy.deepcopy(config.data_config)
            # RS coarsens to roughly 1/3 of the size of the grid, CLJP to roughly 1/2
            fine_data_config.num_unknowns = config.data_config.num_unknowns * 2
            fine_run_dataset = create_dataset(config.train_config.samples_per_run,
                                              fine_data_config,
                                              run=run,
                                              matlab_engine=matlab_engine)
            coarse_run_dataset = create_coarse_dataset(fine_run_dataset, old_model,
                                                       config.data_config,
                                                       config.run_config,
                                                       matlab_engine=matlab_engine)

            combined_run_dataset = run_dataset + coarse_run_dataset
            combined_run_dataset = combined_run_dataset.shuffle()

            checkpoint = train_run(combined_run_dataset, run, batch_size, config,
                                   model, optimizer, global_step,
                                   checkpoint_prefix,
                                   eval_dataset, eval_A_graphs_tuple, eval_config,
                                   matlab_engine)
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=tf_config)

    fire.Fire(train)
