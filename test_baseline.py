import fire
import numpy as np
from pyamg import ruge_stuben_solver, smoothed_aggregation_solver, rootnode_solver
from tqdm import tqdm

import configs
from cr_solver import cr_solver
from data import generate_A


def test_size(size, test_config):
    baseline_errors_div_diff = []
    operator_complexities = []

    fp_threshold = test_config.fp_threshold
    strength = test_config.strength
    presmoother = test_config.presmoother
    postsmoother = test_config.postsmoother
    coarse_solver = test_config.coarse_solver

    cycle = test_config.cycle
    splitting = test_config.splitting
    num_runs = test_config.num_runs
    dist = test_config.dist
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

        baseline_residuals = []

        if splitting is 'CR' or splitting[0] is 'CR':
            baseline_solver = cr_solver(A,
                                        presmoother=presmoother,
                                        postsmoother=postsmoother,
                                        keep=True, max_levels=max_levels,
                                        CF=splitting,
                                        coarse_solver=coarse_solver)
        elif splitting is 'SA':
            baseline_solver = smoothed_aggregation_solver(A,
                                                          strength=strength,
                                                          presmoother=presmoother,
                                                          postsmoother=postsmoother,
                                                          max_levels=max_levels,
                                                          keep=True,
                                                          coarse_solver=coarse_solver)
        elif splitting is 'rootnode':
            baseline_solver = rootnode_solver(A,
                                              strength=strength,
                                              presmoother=presmoother,
                                              postsmoother=postsmoother,
                                              max_levels=max_levels,
                                              keep=True,
                                              coarse_solver=coarse_solver)
        else:
            baseline_solver = ruge_stuben_solver(A,
                                                 strength=strength,
                                                 interpolation='direct',
                                                 presmoother=presmoother,
                                                 postsmoother=postsmoother,
                                                 keep=True, max_levels=max_levels,
                                                 CF=splitting,
                                                 coarse_solver=coarse_solver)

        operator_complexities.append(baseline_solver.operator_complexity())

        _ = baseline_solver.solve(b, x0=x0, tol=0.0, maxiter=iterations, cycle=cycle,
                                  residuals=baseline_residuals)
        baseline_residuals = np.array(baseline_residuals)
        baseline_residuals = baseline_residuals[baseline_residuals > fp_threshold]
        baseline_factor = baseline_residuals[-1] / baseline_residuals[-2]
        baseline_errors_div_diff.append(baseline_factor)

    baseline_errors_div_diff = np.array(baseline_errors_div_diff)
    baseline_errors_div_diff_mean = np.mean(baseline_errors_div_diff)
    baseline_errors_div_diff_std = np.std(baseline_errors_div_diff)

    operator_complexity_mean = np.mean(operator_complexities)
    operator_complexity_std = np.std(operator_complexities)

    if type(splitting) == tuple:
        splitting_str = splitting[0] + '_' + '_'.join([f'{key}_{value}' for key, value in splitting[1].items()])
    else:
        splitting_str = splitting
    results_file = open(
        f"results/baseline/{dist}_{num_unknowns}_cycle_{cycle}_max_levels_{max_levels}_split_{splitting_str}_results.txt", 'w')
    print(f"cycle: {cycle}, max levels: {max_levels}", file=results_file)

    print(f"asymptotic error factor baseline: {baseline_errors_div_diff_mean:.4f} ± {baseline_errors_div_diff_std:.5f}",
          file=results_file)

    print(f"num unknowns: {num_unknowns}")
    print(f"asymptotic error factor baseline: {baseline_errors_div_diff_mean:.4f} ± {baseline_errors_div_diff_std:.5f}")

    print(f"operator complexity: {operator_complexity_mean:.4f} ± {operator_complexity_std:.5f}")
    print(f"operator complexity: {operator_complexity_mean:.4f} ± {operator_complexity_std:.5f}",
          file=results_file)

    results_file.close()


def test_baseline(config='GRAPH_LAPLACIAN_TEST', seed=1):
    # fix random seeds for reproducibility
    np.random.seed(seed)

    test_config = getattr(configs, config).test_config

    for size in test_config.test_sizes:
        test_size(size, test_config)


if __name__ == '__main__':
    fire.Fire(test_baseline)
