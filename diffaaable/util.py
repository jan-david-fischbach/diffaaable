import scipy.linalg
import numpy as np
import jax.numpy as jnp

def poles(z_j,w_j):
  """
  The poles of a barycentric rational with given nodes and weights.
  Poles lifted by zeros of the nominator are included.
  Thus the values $f_j$ do not contribute and don't need to be provided
  The implementation was modified from `baryrat` to support JAX AD.

  Parameters
  ----------
    z_j : array (m,)
      nodes of the barycentric rational
    w_j : array (m,)
      weights of the barycentric rational

  Returns
  -------
    z_n : array (m-1,)
      poles of the barycentric rational (more strictly zeros of the denominator)
  """
  f_j = np.ones_like(z_j)

  B = np.eye(len(w_j) + 1)
  B[0,0] = 0
  E = np.block([[0, w_j],
                [f_j[:,None], np.diag(z_j)]])
  evals = scipy.linalg.eigvals(E, B)
  return evals[np.isfinite(evals)]

def residues(z_j,f_j,w_j,z_n):
  '''
  Residues for given poles via formula for simple poles
  of quotients of analytic functions.
  The implementation was modified from `baryrat` to support JAX AD.

  Parameters
  ----------
    z_j : array (m,)
      nodes of the barycentric rational
    w_j : array (m,)
      weights of the barycentric rational
    z_n : array (n,)
      poles of interest of the barycentric rational (n<=m-1)

  Returns
  -------
    r_n : array (n,)
      residues of poles `z_n`
  '''

  C_pol = 1.0 / (z_n[:,None] - z_j[None,:])
  N_pol = C_pol.dot(f_j*w_j)
  Ddiff_pol = (-C_pol**2).dot(w_j)
  res = N_pol / Ddiff_pol

  return jnp.nan_to_num(res)
