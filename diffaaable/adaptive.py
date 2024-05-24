import jax.numpy as np
from diffaaable import aaa
import jax
from jax import random
from functools import partial

def _adaptive_aaa(z_k_0:np.ndarray,
                 f:callable,
                 evolutions: int = 2,
                 cutoff: float = None,
                 tol: float = 1e-9,
                 radius: float = None,
                 domain: tuple[complex, complex] = None, #TODO
                 f_dot: callable = None):
  """
  z_k initial z_ks
  """
  collect_tangents = f_dot is not None
  z_k = z_k_0
  max_dist = np.max(np.abs(z_k_0[:, np.newaxis] - z_k_0[np.newaxis, :]))*0.4

  if collect_tangents:
    f_unpartial = f.func
    args, _ = jax.tree.flatten(f)
    args_dot, _ = jax.tree.flatten(f_dot)
    z_k_dot = np.zeros_like(z_k)
    f_k , f_k_dot = jax.jvp(
      f_unpartial, (*args, z_k), (*args_dot, z_k_dot)
    )
  else:
    f_k = f(z_k)
    f_k_dot = np.zeros_like(f_k)

  n_eval = len(f_k)
  if cutoff is None:
    cutoff = 1e10*np.max(np.median(np.abs(f_k)))

  if radius is None:
    radius = 1e-3 * max_dist

  def mask(z_k, f_k, f_k_dot):
    m = np.abs(f_k)<cutoff #filter out values, that have diverged too strongly
    return z_k[m], f_k[m], f_k_dot[m]

  key = random.key(0)
  for i in range(evolutions):
    z_k, f_k, f_k_dot = mask(z_k, f_k, f_k_dot)
    z_j, f_j, w_j, z_n = aaa(z_k, f_k, tol)

    key, subkey = jax.random.split(key)

    distance = np.min(np.abs(z_n[:, np.newaxis] - z_k_0[np.newaxis, :]), axis=-1)
    dist_mask = distance < max_dist

    add_z_k = z_n[dist_mask]
    add_z_k += radius*np.exp(1j*2*np.pi*jax.random.uniform(subkey, add_z_k.shape))

    add_z_k_dot = np.zeros_like(add_z_k)
    if collect_tangents:
      f_k_new , f_k_dot_new = jax.jvp(
        f_unpartial, (*args, add_z_k), (*args_dot, add_z_k_dot)
      )
    else:
      f_k_new = f(add_z_k)
      f_k_dot_new = np.zeros_like(f_k_new)

    n_eval += len(z_n)

    z_k = np.append(z_k, add_z_k)
    f_k = np.append(f_k, f_k_new)
    f_k_dot = np.append(f_k_dot, f_k_dot_new)

  z_k, f_k, f_k_dot = mask(z_k, f_k, f_k_dot)

  if collect_tangents:
    return z_k, f_k, f_k_dot

  # import matplotlib.pyplot as plt
  # plt.scatter(z_k.real, z_k.imag)
  # plt.show()
  # plt.close()

  return z_j, f_j, w_j, z_n

@jax.custom_jvp
def adaptive_aaa(z_k_0:np.ndarray,
                 f:callable,
                 evolutions: int = 2,
                 cutoff: float = None,
                 tol: float = 1e-9,
                 radius: float = None,
                 domain: tuple[complex, complex] = None
                 ):
  """
  z_k initial z_ks
  f jax.tree_util.Partial
    (Partial function, with only positional arguments set and one open argument (z))
  """
  return _adaptive_aaa(z_k_0, f, evolutions, cutoff, tol, radius, domain)

@adaptive_aaa.defjvp
def adaptive_aaa_jvp(primals, tangents):
  z_k_0, f = primals[:2]
  z_dot, f_dot = tangents[:2]

  if np.any(z_dot):
    raise NotImplementedError(
      "Parametrizing the sampling positions z_k is not supported"
    )

  z_k, f_k, f_k_dot = \
    _adaptive_aaa(z_k_0, f, *primals[2:], f_dot=f_dot)

  z_k_dot = np.zeros_like(z_k)

  return jax.jvp(aaa, (z_k, f_k), (z_k_dot, f_k_dot))
