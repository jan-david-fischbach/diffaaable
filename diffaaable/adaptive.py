import jax.numpy as np
from diffaaable import aaa
import jax

def _adaptive_aaa(z_k:np.ndarray,
                 f:callable,
                 evolutions: int = 1,
                 cutoff: float = None,
                 tol: float = 1e-13,
                 f_dot: callable = None):
  """
  z_k initial z_ks
  """
  collect_tangents = f_dot is not None

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

  def mask(z_k, f_k, f_k_dot):
    m = np.abs(f_k)<cutoff #filter out values, that have diverged too strongly
    return z_k[m], f_k[m], f_k_dot[m]

  for i in range(evolutions):
    z_k, f_k, f_k_dot = mask(z_k, f_k, f_k_dot)
    z_j, f_j, w_j, z_n = aaa(z_k, f_k, tol)

    z_n_dot = np.zeros_like(z_n)
    if collect_tangents:
      f_k_new , f_k_dot_new = jax.jvp(
        f_unpartial, (*args, z_n), (*args_dot, z_n_dot)
      )
    else:
      f_k_new = f(z_n)
      f_k_dot_new = np.zeros_like(f_k_new)

    n_eval += len(z_n)

    z_k = np.append(z_k, z_n)
    f_k = np.append(f_k, f_k_new)
    f_k_dot = np.append(f_k_dot, f_k_dot_new)

  z_k, f_k, f_k_dot = mask(z_k, f_k, f_k_dot)

  if collect_tangents:
    return z_k, f_k, f_k_dot
  return z_j, f_j, w_j, z_n

@jax.custom_jvp
def adaptive_aaa(z_k_0:np.ndarray,
                 f:callable):
  """
  z_k initial z_ks
  f jax.tree_util.Partial
    (Partial function, with only positional arguments set and one open argument (z))
  """
  return _adaptive_aaa(z_k_0, f)

@adaptive_aaa.defjvp
def adaptive_aaa_jvp(primals, tangents):
  z_k_0, f = primals[:2]
  z_dot, f_dot = tangents[:2]

  if np.any(z_dot):
    raise NotImplementedError(
      "Parametrizing the sampling positions z_k is not supported"
    )

  z_k, f_k, f_k_dot = \
    _adaptive_aaa(z_k_0, f, f_dot=f_dot)

  z_k_dot = np.zeros_like(z_k)

  return jax.jvp(aaa, (z_k, f_k), (z_k_dot, f_k_dot))
