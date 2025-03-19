import numpy as np
import scipy.sparse as sp
import logging

log = logging.getLogger(__name__)

def fast_aaa(z_k, f_k, tol=1e-13, mmax=100):
  """Implementation of the vector valued AAA algorithm avoiding repeated large SVDs

  Args:
      z_k (complex): M sample points
      f_k (complex): MxN array of the sampled vector (size N) at `z_k`
  """

  M = len(z_k)
  N = len(f_k[0])

  norm_f = np.max(np.abs(f_k), axis=0)
  f_k = f_k / norm_f[None, :]

  left_scaling = sp.diags_array(
    f_k.T, offsets=np.arange(0, -M*N, -M), shape=(M*N, M)
  )

  r_k = np.mean(f_k)
  errs = []

  z_j = np.empty((0,))
  index = np.empty((0,), dtype=int)
  w_j = np.empty((0,))
  f_j = np.empty((0,N))
  C = np.empty((M,0))
  Q = np.empty((M,N))

  S = np.zeros((mmax, mmax-1))
  H = np.zeros((mmax, mmax-1))

  for m in range(mmax):
    log.debug(f"Next Iteration {m=}")

    # Select next support point where error is largest
    residual = np.abs(f_k-r_k)
    idx_max_residual = np.argmax(residual)
    err = residual.flat[idx_max_residual]
    errs.append(err)

    if err <= tol:
      break

    next_sample = idx_max_residual % M
    next_sample_flat = next_sample + np.arange(0, N*M, M)

    # Bookkeeping
    index = np.r_[index, next_sample]
    z_j = np.r_[z_j, z_k[next_sample]]
    f_j = np.r_[f_j, [f_k[next_sample]]]

    # Add column to the Cauchy matrix.
    addC = 1/(z_k[:]-z_j[-1])
    C = np.c_[C, addC]
    C[index, m] = 0

    log.debug(f"Next Sample selected: {next_sample}")
    log.debug(f"{z_j.shape=}")
    log.debug(f"{f_j.shape=}")
    log.debug(f"{C.shape=}")


    log.debug(f"{f_j[-1].T.shape=}")
    log.debug(f"{C[:, -1].shape}")

    # "Compute the next vector of the basis."
    v = np.outer(C[:, -1], f_j[-1])
    v = left_scaling @ C[:, -1] - v.reshape(-1, 1)

    q = Q[next_sample_flat, :m-1]
    q = q*S[:m-1, :m-1]

    ee = np.eye(m-1, m-1) - np.conj(q.T) @ q
    np.fill_diagonal(ee, np.real(np.diag(ee)))

    Si = np.linalg.cholesky(ee, upper=True)
    H[:m-1, :m-1] = Si@H[:m-1, :m-1]
    S[:m-1, :m-1] = S@np.linalg.inv(Si)
    S[m, m] = 1
    Q[index, :] = 0

    nv = np.linalg.norm(v)
    H[:m-1, m] = np.conj(Q.T) @ v
    H[:m-1, m] = np.conj(S[:m-1, :m-1].T) @ H[:m-1, m]

    ## got to line 97 of fast_aaa.m


  log.debug(f"Done")
  return z_j, f_j, w_j


################### Testing #################################

def f_test(z, residues):
  return residues/(z[:, None]-(1+1j))

if __name__ == "__main__":

  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
  residues = np.array([1, 3j, 1+2j, -4, 6, 9, 1-3j])
  z_k = np.linspace(-4, 4, 1000) + 0.5j
  z_j, f_j, w_j = fast_aaa(z_k, f_test(z_k, residues), mmax=3)
