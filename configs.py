class DataConfig:
    def __init__(self, dist='lognormal_laplacian_periodic', block_periodic=True,
                 num_unknowns=8 ** 2, root_num_blocks=4, splitting='CLJP', add_diag=False,
                 load_data=True, save_data=False):
        self.dist = dist  # see function 'generate_A()' for possible distributions
        self.block_periodic = block_periodic
        self.num_unknowns = num_unknowns
        self.root_num_blocks = root_num_blocks
        self.splitting = splitting
        self.add_diag = add_diag
        self.load_data = load_data
        self.save_data = save_data


class ModelConfig:
    def __init__(self, mp_rounds=3, global_block=False, latent_size=64, mlp_layers=4, concat_encoder=True):
        self.mp_rounds = mp_rounds
        self.global_block = global_block
        self.latent_size = latent_size
        self.mlp_layers = mlp_layers
        self.concat_encoder = concat_encoder


class RunConfig:
    def __init__(self, node_indicators=True, edge_indicators=True, normalize_rows=True, normalize_rows_by_node=False):
        self.node_indicators = node_indicators
        self.edge_indicators = edge_indicators
        self.normalize_rows = normalize_rows
        self.normalize_rows_by_node = normalize_rows_by_node


class TestConfig:
    def __init__(self, dist='lognormal_laplacian_periodic', splitting='CLJP',
                 test_sizes=(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072),
                 load_data=True, num_runs=100, cycle='W',
                 max_levels=12, iterations=81, fp_threshold=1e-10, strength=('classical', {'theta': 0.25}),
                 presmoother=('gauss_seidel', {'sweep': 'forward', 'iterations': 1}),
                 postsmoother=('gauss_seidel', {'sweep': 'forward', 'iterations': 1}),
                 coarse_solver='pinv2'):
        self.dist = dist
        self.splitting = splitting
        self.test_sizes = test_sizes
        self.load_data = load_data
        self.num_runs = num_runs
        self.cycle = cycle
        self.max_levels = max_levels
        self.iterations = iterations
        self.fp_threshold = fp_threshold
        self.strength = strength
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.coarse_solver = coarse_solver
        # self.coarse_solver = ('gauss_seidel', {'iterations': 20})


class TrainConfig:
    def __init__(self, samples_per_run=256, num_runs=1000, batch_size=32, learning_rate=3e-3, fourier=True,
                 coarsen=False, checkpoint_dir='./training_dir', tensorboard_dir='./tb_dir', load_model=False):
        self.samples_per_run = samples_per_run
        self.num_runs = num_runs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fourier = fourier
        self.coarsen = coarsen
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.load_model = load_model


class Config:
    def __init__(self):
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.run_config = RunConfig()
        self.test_config = TestConfig()
        self.train_config = TrainConfig()


GRAPH_LAPLACIAN_TEST = Config()

COMPLEX_FEM_TEST = Config()
COMPLEX_FEM_TEST.test_config.dist = 'lognormal_complex_fem'
COMPLEX_FEM_TEST.test_config.fp_threshold = 0

GRAPH_LAPLACIAN_RS_TEST = Config()
GRAPH_LAPLACIAN_RS_TEST.test_config.splitting = 'RS'

GRAPH_LAPLACIAN_RS_SECOND_PASS_TEST = Config()
GRAPH_LAPLACIAN_RS_SECOND_PASS_TEST.test_config.splitting = ('RS', {'second_pass': True})

GRAPH_LAPLACIAN_PMIS_TEST = Config()
GRAPH_LAPLACIAN_PMIS_TEST.test_config.splitting = 'PMIS'

GRAPH_LAPLACIAN_PMISc_TEST = Config()
GRAPH_LAPLACIAN_PMISc_TEST.test_config.splitting = 'PMISc'

GRAPH_LAPLACIAN_CLJPc_TEST = Config()
GRAPH_LAPLACIAN_CLJPc_TEST.test_config.splitting = 'CLJPc'

GRAPH_LAPLACIAN_SA_TEST = Config()
GRAPH_LAPLACIAN_SA_TEST.test_config.splitting = 'SA'

GRAPH_LAPLACIAN_ROOTNODE_TEST = Config()
GRAPH_LAPLACIAN_ROOTNODE_TEST.test_config.splitting = 'rootnode'


GRAPH_LAPLACIAN_TRAIN = Config()
GRAPH_LAPLACIAN_TRAIN.data_config.dist = 'lognormal_laplacian_periodic'

GRAPH_LAPLACIAN_ABLATION_MP2 = Config()
GRAPH_LAPLACIAN_ABLATION_MP2.data_config.dist = 'lognormal_laplacian_periodic'
GRAPH_LAPLACIAN_ABLATION_MP2.model_config.mp_rounds = 2

GRAPH_LAPLACIAN_ABLATION_MLP2 = Config()
GRAPH_LAPLACIAN_ABLATION_MLP2.data_config.dist = 'lognormal_laplacian_periodic'
GRAPH_LAPLACIAN_ABLATION_MLP2.model_config.mlp_layers = 2

GRAPH_LAPLACIAN_ABLATION_NO_CONCAT = Config()
GRAPH_LAPLACIAN_ABLATION_NO_CONCAT.data_config.dist = 'lognormal_laplacian_periodic'
GRAPH_LAPLACIAN_ABLATION_NO_CONCAT.model_config.concat_encoder = False

GRAPH_LAPLACIAN_ABLATION_NO_INDICATORS = Config()
GRAPH_LAPLACIAN_ABLATION_NO_INDICATORS.data_config.dist = 'lognormal_laplacian_periodic'
GRAPH_LAPLACIAN_ABLATION_NO_INDICATORS.run_config.node_indicators = False
GRAPH_LAPLACIAN_ABLATION_NO_INDICATORS.run_config.edge_indicators = False

GRAPH_LAPLACIAN_EVAL = Config()
GRAPH_LAPLACIAN_EVAL.data_config.block_periodic = False
GRAPH_LAPLACIAN_EVAL.data_config.num_unknowns = 4096
GRAPH_LAPLACIAN_EVAL.data_config.dist = 'lognormal_laplacian'
GRAPH_LAPLACIAN_EVAL.data_config.load_data = False
GRAPH_LAPLACIAN_EVAL.train_config.fourier = False

SPEC_CLUSTERING_TRAIN = Config()
SPEC_CLUSTERING_TRAIN.data_config.dist = 'spectral_clustering'
SPEC_CLUSTERING_TRAIN.data_config.num_unknowns = 1024
SPEC_CLUSTERING_TRAIN.data_config.block_periodic = False
SPEC_CLUSTERING_TRAIN.data_config.add_diag = True
SPEC_CLUSTERING_TRAIN.train_config.coarsen = False
SPEC_CLUSTERING_TRAIN.train_config.fourier = False

SPEC_CLUSTERING_EVAL = Config()
SPEC_CLUSTERING_EVAL.data_config.dist = 'spectral_clustering'
SPEC_CLUSTERING_EVAL.data_config.num_unknowns = 4096
SPEC_CLUSTERING_EVAL.data_config.block_periodic = False
SPEC_CLUSTERING_EVAL.data_config.add_diag = True
SPEC_CLUSTERING_EVAL.train_config.coarsen = False
SPEC_CLUSTERING_EVAL.train_config.fourier = False



GRAPH_LAPLACIAN_UNIFORM_TEST = Config()
GRAPH_LAPLACIAN_UNIFORM_TEST.data_config.block_periodic = False
GRAPH_LAPLACIAN_UNIFORM_TEST.data_config.dist = 'uniform_laplacian'

FINITE_ELEMENT_TEST = Config()
FINITE_ELEMENT_TEST.data_config.block_periodic = False
FINITE_ELEMENT_TEST.data_config.dist = 'finite_element'

# should replicate results from "Compatible Relaxation and Coarsening in Algebraic Multigrid" (2009)
CR_TEST = Config()
CR_TEST.data_config.splitting = ('CR', {'verbose': True,
                                        'method': 'habituated',
                                        'nu': 2,
                                        'thetacr': 0.5,
                                        'thetacs': [0.3 ** 2, 0.5],
                                        'maxiter': 20})
# CR_TEST.data_config.dist = 'poisson'
# CR_TEST.data_config.dist = 'aniso'
CR_TEST.data_config.dist = 'lognormal_laplacian'
# CR_TEST.data_config.dist = 'example'
CR_TEST.test_config.num_runs = 10
CR_TEST.test_config.test_sizes = (1024, 2048, 4096, 8192,)
# CR_TEST.test_config.test_sizes = ('airfoil', 'bar', 'knot', 'local_disc_galerkin_diffusion',
#                                   'recirc_flow', 'unit_cube', 'unit_square')
# CR_TEST.test_config.fp_threshold = 0
# CR_TEST.test_config.coarse_solver = ('gauss_seidel', {'iterations': 200})

CR_TEST.test_config.presmoother = ('gauss_seidel', {'sweep': 'forward', 'iterations': 1})
# CR_TEST.test_config.postsmoother = ('gauss_seidel', {'sweep': 'backward', 'iterations': 1})
CR_TEST.test_config.coarse_solver = 'pinv2'
CR_TEST.test_config.cycle = 'V'
CR_TEST.test_config.iterations = 40
CR_TEST.test_config.max_levels = 2
