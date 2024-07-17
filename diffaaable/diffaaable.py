from jax import config
config.update("jax_enable_x64", True) #important -> else aaa fails
import jax.numpy as jnp
import jax
import numpy as np
from baryrat import aaa as oaaa # ordinary aaa
import functools
import scipy.linalg

@functools.wraps(oaaa)
@jax.custom_jvp
def aaa(z_k, f_k, tol=1e-13, mmax=100):
  """
  Wraped aaa to enable JAX based autodiff.
  """
  r = oaaa(z_k, f_k, tol=tol, mmax=mmax)
  z_j = r.nodes
  f_j = r.values
  w_j = r.weights
  z_n = poles(z_j, w_j)

  z_n = z_n[jnp.argsort(-jnp.abs(z_n))]

  return z_j, f_j, w_j, z_n

aaa.__doc__ = f"This is a wrapped version of `aaa` as provided by `baryrat`, providing a custom jvp to enable differentiability. For detailed information on the usage of `aaa` please refer to the original documentation: {aaa.__doc__}"

@aaa.defjvp
def aaa_jvp(primals, tangents):
  """
  Derivatives according to [1]: https://arxiv.org/pdf/2403.19404
  Hints for ease of understanding the code:
  
  """
  z_k_full, f_k = primals[:2]
  z_dot, f_dot = tangents[:2]

  primal_out = aaa(z_k_full, f_k)
  z_j, f_j, w_j, z_n = primal_out

  chosen = np.isin(z_k_full, z_j)

  z_k = z_k_full[~chosen]
  f_k = f_k[~chosen]

  # z_dot should be zero anyways
  if np.any(z_dot):
    raise NotImplementedError("Parametrizing the sampling positions z_k is not supported")
  z_k_dot = z_dot[~chosen]
  f_k_dot = f_dot[~chosen] # $\del f_k / \del p$

  ##################################################
  # We have to track which f_dot corresponds to z_k
  sort_orig = jnp.argsort(jnp.abs(z_k_full[chosen]))
  sort_out = jnp.argsort(jnp.argsort(jnp.abs(z_j)))

  z_j_dot = z_dot[chosen][sort_orig][sort_out]
  f_j_dot = f_dot[chosen][sort_orig][sort_out]
  ##################################################

  C = 1/(z_k[:, None]-z_j[None, :]) # Cauchy matrix k x j

  d = C @ w_j # denominator in barycentric formula
  via_f_j = C @ (f_j_dot * w_j) / d # $\sum_j f_j^\prime \frac{\del r}{\del f_j}$

  A = (f_j[None, :] - f_k[:, None])*C/d[:, None]
  b = f_k_dot - via_f_j

  # make sure system is not underdetermined according to eq. 5 of [1]
  A = jnp.concatenate([A, np.conj(w_j.reshape(1, -1))]) 
  b = jnp.append(b, 0)

  with jax.disable_jit(): #otherwise backwards differentiation led to error
    w_j_dot, _, _, _ = jnp.linalg.lstsq(A, b)

  z_n_dot = (
    jnp.sum(w_j_dot.reshape(-1, 1)/(z_n.reshape(1, -1)-z_j.reshape(-1, 1)),    axis=0)/
    jnp.sum(w_j.reshape(-1, 1)    /(z_n.reshape(1, -1)-z_j.reshape(-1, 1))**2, axis=0)
  )

  tangent_out = z_j_dot, f_j_dot, w_j_dot, z_n_dot

  return primal_out, tangent_out

def poles(z_j,w_j):
  """from baryrat"""
  f_j = np.ones_like(z_j)

  B = np.eye(len(w_j) + 1)
  B[0,0] = 0
  E = np.block([[0, w_j],
                [f_j[:,None], np.diag(z_j)]])
  evals = scipy.linalg.eigvals(E, B)
  return evals[np.isfinite(evals)]

def residues(z_j,f_j,w_j,z_n):
  '''residues via formula for simple poles
  of quotients of analytic functions'''

  C_pol = 1.0 / (z_n[:,None] - z_j[None,:])
  N_pol = C_pol.dot(f_j*w_j)
  Ddiff_pol = (-C_pol**2).dot(w_j)
  res = N_pol / Ddiff_pol

  return jnp.nan_to_num(res)
