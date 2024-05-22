import jax.numpy as np
from diffaaable import aaa
import jax

def _adaptive_aaa(z_k:np.ndarray, 
                 f:callable, 
                 arg: np.ndarray,
                 evolutions: int = 3, 
                 cutoff: float = None,
                 tol: float = 1e-13):
  """
  z_k initial z_ks 
  """
  f_k = f(z_k, arg)
  n_eval = len(f_k)
  if cutoff is None:
    cutoff = 1e10*np.max(np.abs(f_k))

  for i in range(evolutions):
    z_j, f_j, w_j, z_n = aaa(z_k, f_k, tol)
    z_k = np.append(z_k, z_n)
    f_k = np.append(f_k, f(z_n, arg))
    n_eval += len(z_n)
    mask = np.abs(f_k)<cutoff #filter out values, that have diverged too strongly
    f_k = f_k[mask]
    z_k = z_k[mask]

  return z_j, f_j, w_j, z_n, z_k, f_k

@jax.custom_vjp
def adaptive_aaa(z_k:np.ndarray, 
                 f:callable, 
                 arg: np.ndarray):
  """
  z_k initial z_ks 
  """
  z_j, f_j, w_j, z_n, z_k, f_k = \
    _adaptive_aaa(z_k, f, arg)
  return z_j, f_j, w_j, z_n

# @adaptive_aaa.defjvp
# def adaptive_aaa_jvp(primals, tangents):
#   print(primals)
#   print(tangents)

#   z_k, f, args, evolutions, cutoff, tol = primals
#   z_dot, f_dot, args_dot = tangents[:3]

#   primals_out = adaptive_aaa(*primals)

#   return primals_out, (z_dot, z_dot, z_dot, z_dot)
  
def f_fwd(z_k:np.ndarray, 
          f:callable, 
          arg: np.ndarray):

  z_j, f_j, w_j, z_n, z_k, f_k = \
    _adaptive_aaa(z_k, f, arg)
  return (z_j, f_j, w_j, z_n), (z_j, f_j, w_j, z_n, z_k, f_k)

def f_bwd(res, g):
  z_j, f_j, w_j, z_n, z_k, f_k = res
  print(z_j, f_j, w_j, z_n, z_k, f_k)
  return ()

adaptive_aaa.defvjp(f_fwd, f_bwd)