import json
import shutil
import os
import glob
from functools import lru_cache

import numpy as np
from collections import Counter


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def most_frequent_splitting(splittings):
    """Given a list of numpy array, returns the most frequent one"""
    list_of_tuples = [tuple(splitting) for splitting in splittings]  # we need a list of immutable types
    counter = Counter(list_of_tuples)
    most_frequent_tuple = counter.most_common(1)[0][0]
    return np.array(most_frequent_tuple)


def create_results_dir(run_name):
    results_dir = 'results/' + run_name
    os.makedirs(results_dir)

    # make a copy of all Python files, for reproducibility
    local_dir = os.path.dirname(__file__)
    for py_file in glob.glob(local_dir + '/*.py'):
        shutil.copy(py_file, results_dir)


def write_config_file(run_name, config, seed):
    results_dir = 'results/' + run_name
    config_dict = {'train_config': config.train_config.__dict__,
                   'data_config': config.data_config.__dict__,
                   'model_config': config.model_config.__dict__,
                   'run_config': config.run_config.__dict__,
                   'seed': seed}
    with open(f'{results_dir}/config.json', 'w') as outfile:
        json.dump(config_dict, outfile)


@lru_cache(maxsize=None)
def tril_indices(grid_size):
    """Cached version of np.tril_indices used for creating relaxation matrices"""
    return np.tril_indices(grid_size)