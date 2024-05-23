import jax.numpy as np
from diffaaable import aaa
import jax

def _adaptive_aaa(z_k:np.ndarray, 
                 f:callable, 
                 evolutions: int = 3, 
                 cutoff: float = None,
                 tol: float = 1e-13):
  """
  z_k initial z_ks 
  """
  f_k = f(z_k)
  n_eval = len(f_k)
  if cutoff is None:
    cutoff = 1e10*np.max(np.median(np.abs(f_k)))

  def mask(z_k, f_k):
    m = np.abs(f_k)<cutoff #filter out values, that have diverged too strongly
    return z_k[m], f_k[m]

  for i in range(evolutions):
    jax.debug.print("f_k: {}", f_k)
    z_k, f_k = mask(z_k, f_k)
    #jax.debug.print("{}: {} -> {}", i, z_k, f_k)
    z_j, f_j, w_j, z_n = aaa(z_k, f_k, tol)
    z_k = np.append(z_k, z_n)
    f_k = np.append(f_k, f(z_n))
    n_eval += len(z_n)

  z_k, f_k = mask(z_k, f_k)
  return z_j, f_j, w_j, z_n, z_k, f_k

@jax.custom_jvp
def adaptive_aaa(z_k:np.ndarray, 
                 f:callable):
  """
  z_k initial z_ks 
  f jax.tree_util.Partial 
    (Partial function, with only positional arguments set and one open argument (z))
  """
  z_j, f_j, w_j, z_n, z_k, f_k = \
    _adaptive_aaa(z_k, f)
  return z_j, f_j, w_j, z_n

@adaptive_aaa.defjvp
def adaptive_aaa_jvp(primals, tangents):
  z_k, f = primals[:2]
  z_dot, f_dot = tangents[:2]

  if np.any(z_dot):
    raise NotImplementedError(
      "Parametrizing the sampling positions z_k is not supported"
    )
  
  f_unpartial = f.func
  args, _ = jax.tree.flatten(f)
  args_dot, _ = jax.tree.flatten(f_dot)

  z_j, f_j, w_j, z_n, z_k_last, f_k_last = \
    _adaptive_aaa(z_k, f)
  
  z_k_last_dot = np.zeros_like(z_k_last)

  # NOTE: the following will perform a redundant evaluation of the primal value 
  # of 'f' to get the gradients TODO: get rid of the redundancy (or cache)
  f_k_last , f_k_last_dot = jax.jvp( 
    f_unpartial, (*args, z_k_last), (*args_dot, z_k_last_dot)
  )
  
  # jax.debug.print("primals -> aaa {}", f_k_last)
  # jax.debug.print("tangents -> aaa {}", f_k_last_dot)

  primals_out, tangents_out = jax.jvp(aaa, (z_k_last, f_k_last), (z_k_last_dot, f_k_last_dot))

  # jax.debug.print("primals out {}", primals_out)
  # jax.debug.print("tangents out {}", tangents_out)
  return primals_out, tangents_out