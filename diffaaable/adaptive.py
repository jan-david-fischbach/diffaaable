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
    cutoff = 1e10*np.max(np.abs(f_k))

  for i in range(evolutions):
    z_j, f_j, w_j, z_n = aaa(z_k, f_k, tol)
    z_k = np.append(z_k, z_n)
    f_k = np.append(f_k, f(z_n))
    n_eval += len(z_n)
    mask = np.abs(f_k)<cutoff #filter out values, that have diverged too strongly
    f_k = f_k[mask]
    z_k = z_k[mask]

  return z_j, f_j, w_j, z_n, z_k, f_k

@jax.custom_jvp
def adaptive_aaa(z_k:np.ndarray, 
                 f:callable):
  """
  z_k initial z_ks 
  """
  z_j, f_j, w_j, z_n, z_k, f_k = \
    _adaptive_aaa(z_k, f)
  return z_j, f_j, w_j, z_n

@adaptive_aaa.defjvp
def adaptive_aaa_jvp(primals, tangents):
  z_k, f = primals[:2]
  z_dot, f_dot = tangents[:2]

  if np.any(z_dot):
    raise NotImplementedError("Parametrizing the sampling positions z_k is not supported")

  z_j, f_j, w_j, z_n, z_k_last, f_k = \
    _adaptive_aaa(z_k, f)


  jax.debug.print("f_dot {}", f_dot)
  tangents_in = f_dot(z_k_last)
  jax.debug.print("poles {}", z_n)
  jax.debug.print("samples {}", z_k_last)
  jax.debug.print("tangents in {}", tangents_in)

  primals_out, tangents_out = jax.jvp(aaa, (z_k, f_k), (np.zeros_like(z_k), tangents_in))

  jax.debug.print("primals out {}", primals_out)
  jax.debug.print("tangents out {}", tangents_out)
  return primals_out, tangents_out
  
# def f_fwd(z_k:np.ndarray, 
#           f:callable):

#   z_j, f_j, w_j, z_n, z_k, f_k = \
#     _adaptive_aaa(z_k, f)
#   return (z_j, f_j, w_j, z_n), (z_j, f_j, w_j, z_n, z_k, f_k)

# def f_bwd(res, g):
#   z_j, f_j, w_j, z_n, z_k, f_k = res
#   print(g)
#   return ()

# adaptive_aaa.defvjp(f_fwd, f_bwd)