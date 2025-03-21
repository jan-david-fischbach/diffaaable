import numpy as np
import scipy.sparse as sp
import logging

log = logging.getLogger(__name__)

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

def set_aaa(z_k, f_k, tol=1e-13, mmax=100, reortho_iterations=10):
  """Implementation of the vector valued AAA algorithm avoiding repeated large SVDs

  Args:
      z_k (complex): M sample points
      f_k (complex): MxN array of the sampled vector (size N) at `z_k`
  """

  M = len(z_k)
  #mmax = min(M//2+1, mmax)
  N = len(f_k[0])

  norm_f = np.max(np.abs(f_k), axis=0)[None, :]
  f_k = f_k / norm_f

  # log.warning(f"{f_k.T.shape=}")
  # left_scaling = sp.diags_array(
  #   f_k.T, offsets=np.arange(0, -M*N, -M), shape=(M*N, M)
  # )
  # log.warning(f"Size: {N}")

  left_scaling = sp.spdiags(
    f_k.T, np.arange(0, -M*N, -M), M*N, M
  )

  r_k = np.mean(f_k)
  errs = []

  z_j = np.empty((0,))
  index = np.empty((0,), dtype=int)
  w_j = np.empty((0,))
  f_j = np.empty((0,N))
  C = np.empty((M,0))
  Q = np.empty((M*N, 0))

  S = np.zeros((mmax+1, mmax+1), dtype=complex)
  H = np.zeros((mmax+1, mmax+1), dtype=complex)

  for m in range(mmax):
    # log.debug(f"Next Iteration {m=}")

    # Select next support point where error is largest
    residual = np.abs(f_k-r_k)

    # log.debug(f"{residual=}")
    
    idx_max_residual = np.argmax(residual)
    err = residual.flat[idx_max_residual]
    errs.append(err)

    log.info(f"Error: {err}")

    if err <= tol:
      break

    next_sample = np.argmax(np.max(residual, axis=1))
    # all indices of the next sample in the flattened f_k array
    next_sample_flat = next_sample + np.arange(0, N*M, M) 

    # log.debug(f"{next_sample=}")

    # Book keeping
    index = np.r_[index, next_sample]
    index_flat = (index[None, :] + np.arange(0, N*M, M)[:, None]).flatten()
    z_j = np.r_[z_j, z_k[next_sample]]   # mx1
    f_j = np.r_[f_j, [f_k[next_sample]]] # mxN

    # Add column to the Cauchy matrix. Mxm
    addC = 1/(z_k[:]-z_j[-1])
    C = np.c_[C, addC]
    C[index, m] = 0

    # log.debug(f"Next Sample selected: {next_sample}")
    # log.debug(f"{z_j.shape=}")
    # log.debug(f"{f_j.shape=}")
    # log.debug(f"{C.shape=}")

    # "Compute the next vector of the basis."

    v = C[:, -1:] @ f_j[-1:]
    # log.debug(f"before left scaling {v.shape=}")
    v = left_scaling @ C[:, -1:] - v.flatten("F")[:, None]
    # log.debug(f"after left scaling {v.shape=}")
    # log.debug(f"{v=}")

    q = Q[next_sample_flat, :m]
    
    # log.debug(f"{q.shape=}")

    q = q@S[:m, :m]

    ee = np.eye(m, m) - np.conj(q.T) @ q
    np.fill_diagonal(ee, np.real(np.diag(ee)))

    Si = np.linalg.cholesky(ee, upper=True)

    # log.debug(f"{Si.shape=}")
    # log.debug(f"{S.shape=}")
    # log.debug(f"{H[:m, :m].shape=}")

    H[:m, :m] = Si@H[:m, :m]
    S[:m, :m] = S[:m, :m]@np.linalg.inv(Si)
    S[m, m] = 1
    Q[index_flat, :] = 0


    Qv = np.conj(Q.T) @ v
    # log.debug(f"{Qv.shape=}")

    H[:m, m] = np.squeeze(Qv)
    H[:m, m] = np.conj(S[:m, :m].T) @ H[:m, m]

    HH = S[:m, :m]@H[:m, m:m+1]

    # log.debug(f"before QHH: {v.shape=}")
    # log.debug(f"before QHH: {Q.shape=}")
    # log.debug(f"before QHH: {HH.shape=}")
    v = v - (Q@HH)
    H[m,m] = nv = np.linalg.norm(v)

    # log.debug(f"before reortho: {v.shape=}")

    # Reorthoganlization is necessary for higher precision
    it = 0
    while it<reortho_iterations and H[m,m] < 1/np.sqrt(2)*nv / 10:
      h_new = np.conj(S[:m, :m].T)@(np.conj(Q.T)@v)
      v = v - Q@(S[:m, :m]@h_new)
      H[:m, m] = H[:m, m] + h_new
      nv = H[m,m]
      H[m,m] = np.linalg.norm(v)
      it += 1

    if it==reortho_iterations:
      log.warning("Hit maximum reorthogonalization iterations")

    v = v/H[m,m]

    # add v
    # log.debug(f"before appending v {Q.shape=}")
    # log.debug(f"{v.shape=}")
    Q = np.c_[Q, v]
    # log.debug(f"after appending v {Q.shape=}")

    # Solve small least squares problem with H
    u, s, vh = np.linalg.svd(H[:m+1, :m+1])
    w_j = np.conj(vh[-1])

    # log.debug(f"{w_j.shape=}")
    # log.debug(f"{f_k.shape=}")

    # log.debug(f"{C.shape=}")

    # Get the rational approximation
    Nom = C@(w_j[:, None]*f_j)
    Den = C@(w_j[:, None]*np.ones_like(f_j))
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
  return z_j, f_j, w_j


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