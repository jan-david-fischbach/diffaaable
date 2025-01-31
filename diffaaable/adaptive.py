from typing import Union
import jax.numpy as np
import numpy.typing as npt
from diffaaable import aaa
import jax
from jax import random
import matplotlib.pyplot as plt
from functools import partial
Domain = tuple[complex, complex]

def top_right(a: npt.NDArray[complex], b: npt.NDArray[complex]):
  return np.logical_and(a.imag>=b.imag, a.real>=b.real)

def domain_mask(domain: Domain, z_n):
    larger_min = top_right(z_n, domain[0])
    smaller_max = top_right(domain[1], z_n)
    return np.logical_and(larger_min, smaller_max)

@jax.tree_util.Partial
def next_samples(z_n, prev_z_n, z_k, domain: Domain, radius, randkey, tolerance=1e-9, min_samples=0, max_samples=0):
  z_n = z_n[domain_mask(domain, z_n)]
  movement = np.min(np.abs(z_n[:, None]-prev_z_n[None, :]), axis=-1)
  ranking = np.argsort(-movement)
  unstable = movement > tolerance
  if np.sum(unstable) > min_samples:
    if max_samples != 0 and np.sum(unstable)>max_samples:
      z_n_unstable = z_n[ranking[:max_samples]]
    else:
      z_n_unstable = z_n[unstable]
  else:
    z_n_unstable = z_n[ranking[:min_samples]]

  add_z_k = z_n_unstable
  add_z_k += radius*np.exp(1j*2*np.pi*jax.random.uniform(randkey, add_z_k.shape))
  return add_z_k


def heat(poles, samples, mesh, sigma):
  #jax.debug.print("poles: {}", poles)
  f_p = np.nansum(
    np.exp(-np.abs(poles[:, None, None]-mesh[None, :])**2/sigma**2),
    axis=0
  )

  #jax.debug.print("{}", f_p)

  f = f_p / np.nansum(
    sigma**2/np.abs(mesh[None, :]-samples[:, None, None])**2,
    axis=0
  )
  return f

@partial(jax.jit, static_argnames=["resolution", "batchsize"])
def _next_samples_heat(
  poles, prev_poles, samples, domain, radius, randkey, resolution=(101, 101),
  batchsize=1, stop=0.2
  ):

  x = np.linspace(domain[0].real, domain[1].real, resolution[0])
  y = np.linspace(domain[0].imag, domain[1].imag, resolution[1])

  X, Y = np.meshgrid(x,y, indexing="ij")
  mesh = X +1j*Y

  add_samples = np.empty(0, dtype=complex)
  for j in range(batchsize):
    heat_map = heat(poles, np.concat([samples, add_samples]), mesh, sigma=radius)
    next_i = np.unravel_index(np.nanargmax(heat_map), heat_map.shape)

    next = np.where(heat_map[next_i] < stop, np.nan, mesh[next_i])
    add_samples = np.append(add_samples, next)

  return add_samples, X, Y, heat(poles, np.concat([samples]), mesh, sigma=radius)

@jax.tree_util.Partial
def next_samples_heat(
  poles, prev_poles, samples, domain, radius, randkey, resolution=(101, 101),
  batchsize=1, stop=0.2, debug=False, debug_known_poles=None
  ):

  add_samples, X, Y, heat_map = _next_samples_heat(
    poles, prev_poles, samples, domain, radius, randkey, resolution,
    batchsize, stop
  )

  add_samples = add_samples[~np.isnan(add_samples)]

  if debug:
    ax = plt.gca()
    plt.figure()
    plt.title(f"radius={radius}")
    if resolution[1]>1:
      plt.pcolormesh(X, Y, heat_map, vmax=1)#, alpha=np.clip(heat_map, 0, 1))
      plt.colorbar()
      plt.scatter(samples.real, samples.imag, label="samples", zorder=1)
      plt.scatter(poles.real, poles.imag, color="C1", marker="x", label="est. pole", zorder=2)
      plt.scatter(add_samples.real, add_samples.imag, color="C2", label="next samples", zorder=3)
      if debug_known_poles is not None:
        plt.scatter(debug_known_poles.real, debug_known_poles.imag, color="C3", marker="+", label="known pole")
      plt.xlim(domain[0].real, domain[1].real)
      plt.ylim(domain[0].imag, domain[1].imag)
      plt.legend(loc="lower right")
    else:
      plt.plot(np.squeeze(X), np.squeeze(heat_map))
      plt.scatter(samples.real, np.zeros(len(samples)))
      plt.scatter(poles.real, np.zeros(len(poles)), color="C1", marker="x", label="est. pole", zorder=2)
      plt.xlim(domain[0].real, domain[1].real)
    plt.savefig(f"{debug}/{len(samples)}.png")
    plt.close()
    plt.sca(ax)

  return add_samples

aaa = jax.tree_util.Partial(aaa)

def mask(z_k, f_k, f_k_dot, cutoff):
    m = np.abs(f_k)<cutoff    #filter out values, that have diverged too strongly
    m = np.logical_and(m, ~np.isnan(f_k))   #filter out nans
    m = np.logical_and(m, ~np.isnan(z_k))   #filter out nans

    if m.ndim == 2:
      m = np.all(m, axis=1)
    z_k, f_k, f_k_dot = z_k[m], f_k[m], f_k_dot[m]

    z_k, idx = np.unique(z_k, return_index=True) #filter out duplicates
    return z_k, f_k[idx], f_k_dot[idx]

def _adaptive_aaa(z_k_0: npt.NDArray,
                 f: callable,
                 evolutions: int = 2,
                 cutoff: float = None,
                 tol: float = 1e-9,
                 mmax: int = 100,
                 radius: float = None,
                 domain: tuple[complex, complex] = None,
                 f_k_0: npt.NDArray = None,
                 sampling: Union[callable, str] = next_samples,
                 prev_z_n: npt.NDArray = None,
                 return_samples: bool = False,
                 aaa: callable = aaa,
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

  if sampling == "heat":
    sampling = next_samples_heat

  collect_tangents = f_dot is not None
  z_k = z_k_0
  max_dist = np.max(np.abs(z_k_0[:, np.newaxis] - z_k_0[np.newaxis, :]))

  if collect_tangents:
    f_unpartial = f.func
    args, _ = jax.tree.flatten(f)
    args_dot, _ = jax.tree.flatten(f_dot)
    z_k_dot = np.zeros_like(z_k)
    f_k, f_k_dot = jax.jvp(
      f_unpartial, (*args, z_k), (*args_dot, z_k_dot)
    )
  else:
    if f_k_0 is None:
      f_k = f(z_k)
    else:
      f_k = f_k_0
    f_k_dot = np.zeros_like(f_k)

  if cutoff is None:
    cutoff = 1e10*np.nanmedian(np.abs(f_k))

  if domain is None:
    center = np.mean(z_k)
    disp = max_dist*(1+1j)
    domain = (center-disp, center+disp)

  if radius is None:
    radius = 1e-3 * max_dist

  if prev_z_n is None:
    prev_z_n = np.array([np.inf], dtype=complex)

  key = random.key(0)
  for i in range(evolutions):
    z_k, f_k, f_k_dot = mask(z_k, f_k, f_k_dot, cutoff=cutoff)
    z_j, f_j, w_j, z_n = aaa(z_k, f_k, tol, mmax)

    if i==evolutions-1:
      break

    key, subkey = jax.random.split(key)
    #print(f"{z_n=}")
    #print(f"{prev_z_n=}")
    add_z_k = sampling(z_n, prev_z_n, z_k, domain, radius, subkey)
    prev_z_n = z_n
    add_z_k_dot = np.zeros_like(add_z_k)

    if len(add_z_k) == 0:
      break

    if collect_tangents:
      f_k_new , f_k_dot_new = jax.jvp(
        f_unpartial, (*args, add_z_k), (*args_dot, add_z_k_dot)
      )
    else:
      f_k_new = f(add_z_k)
      f_k_dot_new = np.zeros_like(f_k_new)

    z_k = np.append(z_k, add_z_k)
    f_k = np.concatenate([f_k, f_k_new])
    f_k_dot = np.concatenate([f_k_dot, f_k_dot_new])

  z_k, f_k, f_k_dot = mask(z_k, f_k, f_k_dot, cutoff=cutoff)

  if collect_tangents:
    return z_k, f_k, f_k_dot
  if return_samples:
    return z_j, f_j, w_j, z_n, z_k, f_k
  return z_j, f_j, w_j, z_n

@jax.custom_jvp
def adaptive_aaa(z_k_0: npt.NDArray,
                 f:callable,
                 evolutions: int = 2,
                 cutoff: float = None,
                 tol: float = 1e-9,
                 mmax: int = 100,
                 radius: float = None,
                 domain: Domain = None,
                 f_k_0: npt.NDArray = None,
                 sampling: callable = next_samples,
                 prev_z_n: npt.NDArray = None,
                 return_samples: bool = False,
                 aaa: callable = aaa):
  """ An 2x adaptive Antoulas–Anderson algorithm for rational approximation of
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
  f_k_0:
      Allows user to provide f evaluated at z_k_0


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
  return _adaptive_aaa(
    z_k_0=z_k_0, f=f, evolutions=evolutions, cutoff=cutoff, tol=tol, mmax=mmax,
    radius=radius, domain=domain, f_k_0=f_k_0, sampling=sampling,
    prev_z_n=prev_z_n, return_samples=return_samples, aaa=aaa
  )

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
