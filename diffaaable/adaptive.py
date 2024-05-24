import jax.numpy as np
import numpy.typing as npt
from diffaaable import aaa
import jax
from jax import random

def top_right(a: npt.NDArray[complex], b: npt.NDArray[complex]):
  return np.logical_and(a.imag>b.imag, a.real>b.real)

def _adaptive_aaa(z_k_0: npt.NDArray,
                 f: callable,
                 evolutions: int = 2,
                 cutoff: float = None,
                 tol: float = 1e-9,
                 radius: float = None,
                 domain: tuple[complex, complex] = None, #TODO
                 f_dot: callable = None):
  """
  Implementation of `adaptive_aaa`

  Parameters
  ----------
  see `adaptive_aaa`

  f_dot: callable
    Tangent of `f`. If provided JVPs of `f` will be collected throughout the
    iterations. For use in custom_jvp
  """
  collect_tangents = f_dot is not None
  z_k = z_k_0
  max_dist = np.max(np.abs(z_k_0[:, np.newaxis] - z_k_0[np.newaxis, :]))

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
    cutoff = 1e10*np.median(np.abs(f_k))

  if domain is None:
    center = np.mean(z_k)
    disp = max_dist*(1+1j)
    domain = (center-disp, center+disp)

  if radius is None:
    radius = 1e-3 * max_dist

  def mask(z_k, f_k, f_k_dot):
    m = np.abs(f_k)<cutoff #filter out values, that have diverged too strongly
    return z_k[m], f_k[m], f_k_dot[m]

  def domain_mask(z_n):
    larger_min = top_right(z_n, domain[0])
    smaller_max = top_right(domain[1], z_n)
    return np.logical_and(larger_min, smaller_max)

  key = random.key(0)
  for i in range(evolutions):
    z_k, f_k, f_k_dot = mask(z_k, f_k, f_k_dot)
    z_j, f_j, w_j, z_n = aaa(z_k, f_k, tol)

    key, subkey = jax.random.split(key)

    add_z_k = z_n[domain_mask(z_n)]
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
  """ An 2x adaptive  Antoulas–Anderson algorithm for rational approximation of
  meromorphic functions that are costly to evaluate.

  The algorithm iteratively places additional sample points close to estimated
  positions of poles identified during the past iteration. By this refinement
  scheme the number of function evaluations can be reduced.

  It is JAX differentiable wrt. the approximated function `f`, via its other
  arguments besides `z`. `f` should be provided as a `jax.tree_util.Partial`
  with only positional arguments pre-filled!

  Parameters
  ----------
  z_k_0 : np.ndarray
      Array of initial sample points
  f : callable
      function to be approximated. When using gradients `f` should be provided
      as a `jax.tree_util.Partial` with only positional arguments pre-filled.
      Furthermore it should be compatible to `jax.jvp`.
  evolutions: int
      Number of refinement iterations
  cutoff: float
      Maximum absolute value a function evaluation should take
      to be regarded valid. Otherwise the sample point is discarded.
      Defaults to 1e10 times the median of `f(z_k_0)`
  tol: float
      Tolerance used in AAA (see `diffaaable.aaa`)
  radius: float
      Distance from the assumed poles for nex samples
  domain: tuple[complex, complex]
      Tuple of min (lower left) and max (upper right) values defining a
      rectangle in the complex plane. Assumed poles outside of the domain
      will not receive refinement.


  Returns
  -------
  z_j: np.array
      chosen samples
  f_j: np.array
      `f(z_j)`
  w_j: np.array
      Weights of Barycentric Approximation
  z_n: np.array
      Poles of Barycentric Approximation
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
