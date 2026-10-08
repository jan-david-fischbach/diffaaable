import jax
import numpy as np
from diffaaable.util import poles, aaa_jvp_rule, validate_samples
from diffaaable.errors import InputError, convergence_error, mmax_warning

def check_inputs(z_k, f_k):
  validate_samples(z_k, f_k, vector_valued=True)
  f_k = np.array(f_k)
  z_k = np.array(z_k)
  M = z_k.shape[0]

  if f_k.ndim == 1:
    f_k = f_k[:, np.newaxis]

  if f_k.ndim != 2:
    raise InputError(f"f_k should be 1 or 2D, but has shape {f_k.shape}. "
                     "For higher dimensional samples use `tensor_aaa`.")
  V = f_k.shape[1]

  return z_k, f_k, M, V

def vectorial_aaa(z_k, f_k, tol=1e-13, mmax=100, return_errors=False):
  r"""Find a rational approximation to $\mathbf f(z)$ over the points $z_k$ using
  a modified AAA algorithm, as presented in [^4]. Importantly the weights and
  thus also the poles are shared between all entries of $\mathbf f(z)$.

  Parameters
  ----------
      z_k : array (M,):
        M sample points
      f_k : array (M, V):
        vector valued function values
      tol : float
        the approximation tolerance
      mmax : int
        the maximum number of iterations/degree of the resulting approximant
      return_errors : bool
        additionally return the errors of each iteration (not differentiable)

  Returns
  -------
    z_j : array (m,)
      nodes of the barycentric approximant
    f_j : array (m, V)
      values of the barycentric approximant
    w_j : array (m,)
      weights of the barycentric approximant
    z_n : array (m-1,)
      poles of the barycentric approximant
    errors : list
      only if `return_errors`

  The result is JAX differentiable with respect to `f_k` (via a custom JVP
  analogous to `diffaaable.aaa`) unless `return_errors` is set.
  """
  if return_errors:
    return _vectorial_aaa(z_k, f_k, tol, mmax, return_errors=True)
  return _vectorial_aaa_diff(z_k, f_k, tol, mmax)

def _vectorial_aaa(z_k, f_k, tol=1e-13, mmax=100, return_errors=False):
  z_k, f_k, M, V = check_inputs(z_k, f_k)

  J = np.ones(M, dtype=bool)
  z_j = np.empty(0, dtype=z_k.dtype)
  f_j = np.empty((0, V), dtype=f_k.dtype)
  errors = []

  reltol = tol * np.linalg.norm(f_k, np.inf)

  r_k = np.mean(f_k) * np.ones_like(f_k)

  mmax = min(mmax, len(f_k)//2)
  if mmax < 1:
    raise InputError(f"vectorial AAA requires at least 2 samples, got {M}")

  for m in range(mmax):
      # find largest residual
      jj = np.argmax(np.linalg.norm(f_k - r_k, axis=-1)) #Next sample point to include
      z_j = np.append(z_j, np.array([z_k[jj]]))
      f_j = np.concatenate([f_j, f_k[jj][None, :]])
      J[jj] = False

      # Cauchy matrix containing the basis functions as columns
      C = 1.0 / (z_k[J,None] - z_j[None,:])
      # Loewner matrix
      A = (f_k[J,None] - f_j[None,:]) * C[:,:,None]

      # TODO: stack A
      A = np.concatenate(np.moveaxis(A, -1, 0))

      # compute weights as right singular vector for smallest singular value
      if return_errors:
         print("start SVD")
      try:
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
      except np.linalg.LinAlgError as e:
        raise convergence_error(e, "vectorial AAA") from e
      if return_errors:
         print("finished SVD")

      w_j = Vh[-1, :].conj()

      # approximation: numerator / denominator
      N = C.dot(w_j[:, None] * f_j) #TODO check it works
      D = C.dot(w_j)[:, None]

      # update residual
      r_k = f_k.copy()
      r_k[J] = N / D

      # check for convergence
      errors.append(np.linalg.norm(f_k - r_k, np.inf))
      if return_errors:
         print(errors[-1])
      if errors[-1] <= reltol:
          break
  else:
    mmax_warning(mmax, errors[-1] / np.linalg.norm(f_k, np.inf), tol,
                 "vectorial AAA")

  z_n = poles(z_j, w_j)
  if return_errors:
    return z_j, f_j, w_j, z_n, errors
  return z_j, f_j, w_j, z_n


@jax.custom_jvp
def _vectorial_aaa_diff(z_k, f_k, tol=1e-13, mmax=100):
  return _vectorial_aaa(z_k, f_k, tol, mmax)

_vectorial_aaa_diff.defjvp(aaa_jvp_rule(_vectorial_aaa))

def residues_vec(z_j,f_j,w_j,z_n):
  '''Vectorial residues for given poles via formula for simple poles
  of quotients of analytic functions.
  '''

  C_pol = 1.0 / (z_n[:,None] - z_j[None,:])
  N_pol = C_pol.dot((f_j*w_j[None,:]).T)
  Ddiff_pol = (-C_pol**2).dot(w_j)
  res = N_pol / Ddiff_pol[:, None]

  return np.nan_to_num(res.T)
