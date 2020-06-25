# Learning Algebraic Multigrid using Graph Neural Networks
Code for reproducing the experimental results in our paper:
https://arxiv.org/abs/2003.05744

## Requirements
 * Python >= 3.6
 * Tensorflow >= 1.14
 * NumPy
 * PyAMG
 * Graph Nets: https://github.com/deepmind/graph_nets
 * MATLAB >= R2019a
 * MATLAB engine for Python: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    * Requires modifying internals of Python library for efficient passing of NumPy arrays, as described here: https://stackoverflow.com/a/45290997
 * tqdm
 * Fire: https://github.com/google/python-fire
 * scikit-learn
 * Meshpy: https://documen.tician.de/meshpy/index.html
 

## Training
### Graph Laplacian
```
python train.py
```
Model checkpoint is saved at 'training_dir/*model_id*', where *model_id* is a randomly generated 5 digit string.

Tensorboard log files are outputted to 'tb_dir/*model_id*'.

A copy of the .py files and a JSON file that describes the configuration are saved to 'results/*model_id*'.

A random seed can be specified by setting a `-seed` argument.
### Spectral clustering
```
python train.py -config SPEC_CLUSTERING_TRAIN -eval-config SPEC_CLUSTERING_EVAL
```

### Ablation study
```
python train.py -config GRAPH_LAPLACIAN_ABLATION_MLP2
python train.py -config GRAPH_LAPLACIAN_ABLATION_MP2
python train.py -config GRAPH_LAPLACIAN_ABLATION_NO_CONCAT
python train.py -config GRAPH_LAPLACIAN_ABLATION_NO_INDICATORS
```
Other model configurations and hyper-parameters can be trained by creating `Config` objects in `configs.py`, and setting the appropriate `-config` argument.

## Evaluation
### Graph Laplacian lognormal distribution
```
python test_model.py -model-name 12345  
```
Replace `12345` by the *model_id* of a previously trained model.

Results are saved at 'results/*model_id*'.

### Graph Laplacian uniform distribution
```
python test_model.py -model-name 12345 -config GRAPH_LAPLACIAN_UNIFORM_TEST
```

### Finite element
```
python test_model.py -model-name 12345 -config FINITE_ELEMENT_TEST
```

### Spectral clustering
```
python spec_cluster.py -model-name 12345
```