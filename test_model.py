import json
from functools import partial

import fire
import matlab.engine
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import configs
from data import generate_A
from model import get_model
from prolongation_functions import model, baseline
from ruge_stuben_custom_solver import ruge_stuben_custom_solver


def test_size(model_name, graph_model, size, test_config, run_config, matlab_engine):
    model_prolongation = partial(model, graph_model=graph_model,
                                 normalize_rows_by_node=run_config.normalize_rows_by_node,
                                 edge_indicators=run_config.edge_indicators,
                                 node_indicators=run_config.node_indicators,
                                 matlab_engine=matlab_engine)
    baseline_prolongation = baseline

    model_errors_div_diff = []
    baseline_errors_div_diff = []

    fp_threshold = test_config.fp_threshold
    strength = test_config.strength
    presmoother = test_config.presmoother
    postsmoother = test_config.postsmoother
    coarse_solver = test_config.coarse_solver

    cycle = test_config.cycle
    splitting = test_config.splitting
    dist = test_config.dist
    num_runs = test_config.num_runs
    max_levels = test_config.max_levels
    iterations = test_config.iterations
    load_data = test_config.load_data

    block_periodic = False
    root_num_blocks = 1

    if load_data:
        if dist == 'lognormal_laplacian_periodic':
            As = np.load(f"test_data_dir/delaunay_periodic_logn_num_As_{100}_num_points_{size}.npy")
        elif dist == 'lognormal_complex_fem':
            As = np.load(f"test_data_dir/fe_hole_logn_num_As_{100}_num_points_{size}.npy")
        else:
            raise NotImplementedError()

    for i in tqdm(range(num_runs)):
        if load_data:
            A = As[i]
        else:
            A = generate_A(size, dist, block_periodic, root_num_blocks)

        num_unknowns = A.shape[0]
        x0 = np.random.normal(loc=0.0, scale=1.0, size=num_unknowns)
        b = np.zeros((A.shape[0]))

        model_residuals = []
        baseline_residuals = []

        model_solver = ruge_stuben_custom_solver(A, model_prolongation,
                                                 strength=strength,
                                                 presmoother=presmoother,
                                                 postsmoother=postsmoother,
                                                 keep=True, max_levels=max_levels,
                                                 CF=splitting,
                                                 coarse_solver=coarse_solver)

        _ = model_solver.solve(b, x0=x0, tol=0.0, maxiter=iterations, cycle=cycle,
                               residuals=model_residuals)
        model_residuals = np.array(model_residuals)
        model_residuals = model_residuals[model_residuals > fp_threshold]
        model_factor = model_residuals[-1] / model_residuals[-2]
        model_errors_div_diff.append(model_factor)

        baseline_solver = ruge_stuben_custom_solver(A, baseline_prolongation,
                                                    strength=strength,
                                                    presmoother=presmoother,
                                                    postsmoother=postsmoother,
                                                    keep=True, max_levels=max_levels,
                                                    CF=splitting,
                                                    coarse_solver=coarse_solver)

        _ = baseline_solver.solve(b, x0=x0, tol=0.0, maxiter=iterations, cycle=cycle,
                                  residuals=baseline_residuals)
        baseline_residuals = np.array(baseline_residuals)
        baseline_residuals = baseline_residuals[baseline_residuals > fp_threshold]
        baseline_factor = baseline_residuals[-1] / baseline_residuals[-2]
        baseline_errors_div_diff.append(baseline_factor)

    model_errors_div_diff = np.array(model_errors_div_diff)
    baseline_errors_div_diff = np.array(baseline_errors_div_diff)
    model_errors_div_diff_mean = np.mean(model_errors_div_diff)
    model_errors_div_diff_std = np.std(model_errors_div_diff)
    baseline_errors_div_diff_mean = np.mean(baseline_errors_div_diff)
    baseline_errors_div_diff_std = np.std(baseline_errors_div_diff)

    if type(splitting) == tuple:
        splitting_str = splitting[0] + '_'+ '_'.join([f'{key}_{value}' for key, value in splitting[1].items()])
    else:
        splitting_str = splitting
    results_file = open(
        f"results/{model_name}/{dist}_{num_unknowns}_cycle_{cycle}_max_levels_{max_levels}_split_{splitting_str}_results.txt",
        'w')
    print(f"cycle: {cycle}, max levels: {max_levels}", file=results_file)
    print(f"asymptotic error factor model: {model_errors_div_diff_mean:.4f} ± {model_errors_div_diff_std:.5f}",
          file=results_file)

    print(f"asymptotic error factor baseline: {baseline_errors_div_diff_mean:.4f} ± {baseline_errors_div_diff_std:.5f}",
          file=results_file)
    model_success_rate = sum(model_errors_div_diff < baseline_errors_div_diff) / num_runs
    print(f"model success rate: {model_success_rate}",
          file=results_file)

    print(f"num unknowns: {num_unknowns}")
    print(f"asymptotic error factor model: {model_errors_div_diff_mean:.4f} ± {model_errors_div_diff_std:.5f}")
    print(f"asymptotic error factor baseline: {baseline_errors_div_diff_mean:.4f} ± {baseline_errors_div_diff_std:.5f}")
    print(f"model success rate: {model_success_rate}")

    results_file.close()


def test_model(model_name=None, test_config='GRAPH_LAPLACIAN_TEST', seed=1):
    if model_name is None:
        raise RuntimeError("model name required")
    model_name = str(model_name)
    matlab_engine = matlab.engine.start_matlab()

    # fix random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    matlab_engine.eval(f'rng({seed})')

    test_config = getattr(configs, test_config).test_config
    config_file = f"results/{model_name}/config.json"
    with open(config_file) as f:
        data = json.load(f)
        model_config = configs.ModelConfig(**data['model_config'])
        run_config = configs.RunConfig(**data['run_config'])

    model = get_model(model_name, model_config, run_config, matlab_engine)

    for size in test_config.test_sizes:
        test_size(model_name, model, size, test_config, run_config,
                  matlab_engine)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

    fire.Fire(test_model)
