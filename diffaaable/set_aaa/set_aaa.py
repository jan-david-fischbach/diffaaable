import jax
import numpy as np
import logging
from diffaaable.util import poles, aaa_jvp_rule

log = logging.getLogger(__name__)

np.set_printoptions(edgeitems=30, linewidth=100000,
    precision=14)

@jax.custom_jvp
def set_aaa(z_k, f_k, tol=1e-13, mmax=100, reortho_iterations=3, normalize=True):
  """Implementation of the vector valued AAA algorithm avoiding repeated large SVDs

  JAX differentiable with respect to `f_k` via a custom JVP analogous to
  `diffaaable.aaa`.

  Args:
      z_k (complex): M sample points
      f_k (complex): MxN array of the sampled vector (size N) at `z_k`
      tol (float): approximation tolerance
      mmax (int): maximum number of support points
      reortho_iterations (int): maximum number of reorthogonalization steps
      normalize (bool): normalize each entry by its maximum absolute value

  Returns:
      z_j, f_j, w_j, z_n: nodes, values (mxN), weights and poles
  """
  return _set_aaa(z_k, f_k, tol, mmax, reortho_iterations, normalize)

def _set_aaa(z_k, f_k, tol=1e-13, mmax=100, reortho_iterations=3, normalize=True):

  z_k = np.asarray(z_k)
  f_k = np.asarray(f_k)
  M = len(z_k)
  mmax = min(M-1, mmax)
  N = len(f_k[0])

  norm_f = np.max(np.abs(f_k), axis=0)[None, :] if normalize else 1
  f_k = f_k / norm_f

  r_k = np.mean(f_k)
  errs = []

  z_j = np.empty((0,))
  index = np.empty((0,), dtype=int)
  w_j = np.empty((0,))
  f_j = np.empty((0,N))
  C = np.zeros((M, mmax+1), dtype=complex)
  # orthonormal basis, stored column-wise; grown on demand to avoid copying
  # the (potentially huge) M*N x m matrix in every iteration
  Q_buf = np.zeros((M*N, min(mmax+1, 8)), dtype=complex, order='F')

  S = np.zeros((mmax+1, mmax+1), dtype=complex)
  H = np.zeros((mmax+1, mmax+1), dtype=complex)

  for m in range(mmax):

    # Select next support point where error is largest
    residual = np.abs(f_k-r_k)

    idx_max_residual = np.argmax(residual)
    err = residual.flat[idx_max_residual]
    errs.append(err)

    log.info(f"Error: {err}")

    if err <= tol:
      break

    next_sample = np.argmax(np.max(residual, axis=1))
    # all indices of the next sample in the flattened f_k array
    next_sample_flat = next_sample + np.arange(0, N*M, M)

    log.debug(f"next_sample {int(next_sample)}")

    # Book keeping
    index = np.r_[index, next_sample]
    index_flat = (index[None, :] + np.arange(0, N*M, M)[:, None]).flatten()
    z_j = np.r_[z_j, z_k[next_sample]]   # mx1
    f_j = np.r_[f_j, [f_k[next_sample]]] # mxN

    # Add column to the Cauchy matrix. Mxm
    with np.errstate(divide='ignore'):
        C[:, m] = 1/(z_k[:]-z_j[-1])
        C[index, m] = 0
    Cm = C[:, :m+1]

    if m >= Q_buf.shape[1]:
      Q_new = np.zeros((M*N, min(mmax+1, 2*Q_buf.shape[1])), dtype=complex, order='F')
      Q_new[:, :m] = Q_buf[:, :m]
      Q_buf = Q_new
    Q = Q_buf[:, :m]

    # "Compute the next vector of the basis."
    v = (C[:, m:m+1] * (f_k - f_j[-1:])).flatten("F")[:, None]

    q = Q[next_sample_flat, :m]

    q = q@S[:m, :m]

    ee = np.eye(m, m) - np.conj(q.T) @ q
    np.fill_diagonal(ee, np.real(np.diag(ee)))

    Si = np.linalg.cholesky(ee, upper=True)
    #Si = scipy.linalg.cholesky(ee, lower=False)

    H[:m, :m] = Si@H[:m, :m]
    # np.linalg (not scipy.linalg): alternating between the separate BLAS
    # builds bundled with numpy and scipy causes heavy thread contention
    S[:m, :m] = np.linalg.solve(Si.T, S[:m, :m].T).T
    S[m, m] = 1
    Q[index_flat, :] = 0

    Qv = (v.conj().T @ Q).conj().T # = Q^H v without copying Q

    H[:m, m] = np.squeeze(Qv)
    H[:m, m] = np.conj(S[:m, :m].T) @ H[:m, m]

    HH = S[:m, :m]@H[:m, m:m+1]

    nv = np.linalg.norm(v)
    v = v - (Q@HH)
    H[m,m] = np.linalg.norm(v)

    # Reorthoganlization is necessary for higher precision
    it = 0
    while it<reortho_iterations and H[m,m] < 1/np.sqrt(2)*nv:
      h_new = np.conj(S[:m, :m].T)@((v.conj().T @ Q).conj().T)
      v = v - Q@(S[:m, :m]@h_new)
      H[:m, m] = H[:m, m] + np.squeeze(h_new)
      nv = H[m,m]
      H[m,m] = np.linalg.norm(v)
      it += 1

    if it==reortho_iterations:
      log.warning("Hit maximum reorthogonalization iterations")

    v = v/H[m,m]

    # add v
    Q_buf[:, m] = v[:, 0]

    # Solve small least squares problem with H
    u, s, vh = np.linalg.svd(H[:m+1, :m+1])
    w_j = np.conj(vh[-1])

    # Get the rational approximation
    Nom = Cm@(w_j[:, None]*f_j)
    Den = (Cm@w_j)[:, None]
    with np.errstate(invalid='ignore'):
      r_k = Nom/Den
    r_k[index] = f_j




  f_j *= norm_f
  w_j = w_j[:m+1] # not sure this is needed
  errs=np.array(errs)[:, None]*norm_f

  zero_weight_mask = w_j == 0
  z_j = z_j[~zero_weight_mask]
  f_j = f_j[~zero_weight_mask]
  w_j = w_j[~zero_weight_mask]

  log.debug(f"Done with Set AAA")

  z_n = poles(z_j, w_j)
  return z_j, f_j, w_j, z_n


def _entry_weights(z_k, f_k, tol, mmax, reortho_iterations, normalize):
  # mirror the normalization of the fit in the least squares problem of the JVP
  if not normalize:
    return None
  return 1/np.max(np.abs(np.asarray(f_k)), axis=0)

set_aaa.defjvp(aaa_jvp_rule(_set_aaa, _entry_weights))


################### Testing #################################

def f_test(z, residues):
  return residues/(z[:, None]-1j+0.1) + residues/(z[:, None]-2j) + residues/(z[:, None]-3)

if __name__ == "__main__":

  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
  residues = np.array([2, 1j])
  #z_k = np.linspace(-1, 1, 5)
  z_k = np.linspace(-1, 1, 9)
  z_j, f_j, w_j = set_aaa(z_k, f_test(z_k, residues), mmax=5)

  from diffaaable.core import poles
  z_n = poles(z_j, w_j)
  print(len(z_n))
