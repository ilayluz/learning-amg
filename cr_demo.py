import numpy as np
import pyamg
import matplotlib.pyplot as plt

from configs import CR_TEST
from cr_solver import cr_solver

size = 33**2
grid_size = int(np.sqrt(size))
A = pyamg.gallery.poisson((grid_size, grid_size), type='FE', format='csr')
# A = pyamg.gallery.stencil_grid(pyamg.gallery.diffusion_stencil_2d(epsilon=0.01, type='FE'),
#                                (grid_size, grid_size),  format='csr')

xx = np.arange(0, grid_size, dtype=float)
x, y = np.meshgrid(xx, xx)
V = np.concatenate([[x.ravel()], [y.ravel()]], axis=0).T

solver = cr_solver(A,
                   keep=True, max_levels=2,
                   CF=CR_TEST.data_config.splitting)
print(solver)
splitting = solver.levels[0].splitting

C = np.where(splitting == 0)[0]
F = np.where(splitting == 1)[0]
plt.scatter(V[C, 0], V[C, 1], marker='s', s=12,
            color=[232.0 / 255, 74.0 / 255, 39.0 / 255])
plt.scatter(V[F, 0], V[F, 1], marker='s', s=12,
            color=[19.0 / 255, 41.0 / 255, 75.0 / 255])
plt.show()
