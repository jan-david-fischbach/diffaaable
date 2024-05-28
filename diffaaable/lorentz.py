from jax import config
config.update("jax_enable_x64", True) #important -> else aaa fails
import jax.numpy as np
import jax
from diffaaable.vectorial import check_inputs
import jaxopt

def optimal_weights(A, A_hat, stepsize=0.5):
  # Initial guess (from uncorrected A)
  _, _, Vh = np.linalg.svd(A)
  w_j = Vh[-1, :].conj()

  def obj_fun(w):
     w /= np.linalg.norm(w)
     err = (A @ w) + (A_hat @ w.conj())
     return np.linalg.norm(err)

  solver = jaxopt.LBFGS(fun=obj_fun, maxiter=300, tol=1e-13)
  res = solver.run(w_j)
  w_j, state = res
  w_j /= np.linalg.norm(w_j)

  return w_j

def lorentz_aaa(z_k, f_k, tol=1e-9, mmax=100, return_errors=False):
  """
  """
  z_k, f_k, M, V = check_inputs(z_k, f_k)

  J = np.ones(M, dtype=bool)
  z_j = np.empty(0, dtype=z_k.dtype)
  f_j = np.empty((0, V), dtype=f_k.dtype)
  errors = []

  reltol = tol * np.linalg.norm(f_k, np.inf)

  r_k = np.mean(f_k) * np.ones_like(f_k)
  # approx.

  for m in range(mmax):
      # find largest residual
      jj = np.argmax(np.linalg.norm(f_k - r_k, axis=-1)) #Next sample point to include
      z_j = np.append(z_j, np.array([z_k[jj]]))
      f_j = np.concatenate([f_j, f_k[jj][None, :]])
      J = J.at[jj].set(False)

      # Cauchy matrix containing the basis functions as columns
      C = 1.0 / (z_k[J,None] - z_j[None,:])
      # Loewner matrix
      A = (f_k[J,None] - f_j[None,:]) * C[:,:,None]
      A = np.concatenate(np.moveaxis(A, -1, 0))

      # Lorentz Correction
      C_hat = 1.0 / (z_k[J,None] + np.conj(z_j)[None,:])
      # Loewner matrix
      A_hat = (f_k[J,None] - np.conj(f_j)[None,:]) * C_hat[:,:,None]
      A_hat = np.concatenate(np.moveaxis(A_hat, -1, 0))

      w_j = optimal_weights(A, A_hat)

      # approximation: numerator / denominator
      N = C.dot(w_j[:, None] * f_j)
      N_hat = C_hat.dot(np.conj(w_j[:, None] * f_j))

      D = C.dot(w_j)[:, None]
      D_hat = C_hat.dot(np.conj(w_j))[:, None]



      # update approximation
      r_k = f_k.at[J].set((N + N_hat) / (D+D_hat))

      # check for convergence
      errors.append(np.linalg.norm(f_k - r_k, np.inf))
      if errors[-1] <= reltol:
          break

  if V == 1:
    f_j = f_j[:, 0]


  z_j = np.concatenate([z_j, -np.conj(z_j)])
  f_j = np.concatenate([f_j, np.conj(f_j)])
  w_j = np.concatenate([w_j, np.conj(w_j)])

  if return_errors:
    return z_j, f_j, w_j, errors
  return z_j, f_j, w_j
