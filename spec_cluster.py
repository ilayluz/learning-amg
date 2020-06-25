import json
from functools import partial

import fire
import numpy as np
import tensorflow as tf
from scipy.sparse.linalg import lobpcg
from tqdm import tqdm
import matlab.engine

import configs
from data import generate_A_spec_cluster
from model import get_model
from prolongation_functions import model, baseline
from ruge_stuben_custom_solver import ruge_stuben_custom_solver


def precond_test(model_name=None, test_config='GRAPH_LAPLACIAN_TEST', seed=1):
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

    graph_model = get_model(model_name, model_config, run_config, matlab_engine)

    max_levels = 2
    cycle = 'V'
    size = 1000
    num_samples = 100
    num_clusters = 2
    dimension = 2
    gamma = None
    distribution = 'moons'

    model_prolongation = partial(model, graph_model=graph_model, normalize_rows_by_node=False,
                                 matlab_engine=matlab_engine)
    baseline_prolongation = baseline

    model_wins = 0
    base_wins = 0
    ties = 0
    total_model_iters = 0
    total_base_iters = 0
    for _ in tqdm(range(num_samples)):
        A = generate_A_spec_cluster(size, num_clusters, unit_std=True, dim=dimension,
                                    dist=distribution, gamma=gamma, distance=False, n_neighbors=10)

        model_solver = ruge_stuben_custom_solver(A, model_prolongation,
                                                 strength=test_config.strength,
                                                 presmoother=test_config.presmoother,
                                                 postsmoother=test_config.postsmoother,
                                                 CF=test_config.splitting,
                                                 max_levels=max_levels)

        model_precond = model_solver.aspreconditioner(cycle=cycle)

        base_solver = ruge_stuben_custom_solver(A, baseline_prolongation,
                                                strength=test_config.strength,
                                                presmoother=test_config.presmoother,
                                                postsmoother=test_config.postsmoother,
                                                CF=test_config.splitting,
                                                max_levels=max_levels)
        base_precond = base_solver.aspreconditioner(cycle=cycle)

        x0 = np.random.uniform(size=[A.shape[0], num_clusters])
        try:
            _, _, model_res_norms = lobpcg(A, x0, M=model_precond, tol=1.e-12, maxiter=100,
                                           largest=False, retResidualNormsHistory=True)

            _, _, base_res_norms = lobpcg(A, x0, M=base_precond, tol=1.e-12, maxiter=100,
                                          largest=False, retResidualNormsHistory=True)
        except np.linalg.LinAlgError:
            print("error")
            continue

        model_iters = len(model_res_norms)
        base_iters = len(base_res_norms)
        if model_iters < base_iters:
            model_wins += 1
        elif model_iters > base_iters:
            base_wins += 1
        else:
            ties += 1

        total_model_iters += model_iters
        total_base_iters += base_iters

    print(f"model wins: {model_wins}")
    print(f"base wins: {base_wins}")
    print(f"win ratio: {model_wins / (model_wins + base_wins)}")
    print(f"avg model iters: {total_model_iters / num_samples}")
    print(f"avg base iters: {total_base_iters / num_samples}")
    print(f"iters ratio: {total_model_iters / total_base_iters}")


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

    fire.Fire(precond_test)
