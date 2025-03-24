from jax import config
config.update("jax_enable_x64", True)

from baryrat import BarycentricRational
import jax.numpy as np
from diffaaable import set_aaa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diffaaable.set_aaa.set_aaa")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("SET AAA> %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.DEBUG)

def f_test(z, residues, n_poles):
  poles = (0.7+0.9j)**np.arange(n_poles)
  return np.sum(residues.T[None, :, :]/(z[:, None, None]-poles[None, None, :]), axis=-1)

def f_test_harder(z, residues, n_poles):
  poles = (0.1)**np.arange(n_poles)
  return np.sum(residues.T[None, :, :]/(z[:, None, None]-poles[None, None, :]), axis=-1)


z_k = np.linspace(-4, 4, 15) + 0.8j

def test_set_aaa():
  n_poles = 3
  residues = np.arange(n_poles*2).reshape(n_poles, -1)
  f_k = f_test_harder(z_k, residues, n_poles)

  print(f_k)
  z_j, f_j, w_j = set_aaa(z_k, f_k, tol=1e-13, reortho_iterations=3)

  r = BarycentricRational(z_j, f_j[:,0], w_j)
  pol, res_found = r.polres()
  print(f"residues: {res_found}")
  sorter = np.argsort(-np.abs(pol))
  sorted_poles = pol[sorter]
  print(f"{sorted_poles=}")

if __name__ == "__main__":
  # lmax = np.arange(1, 10)
  # N = (2 * lmax * (lmax + 2))**2
  # print(np.stack([lmax, N]))

  test_set_aaa()