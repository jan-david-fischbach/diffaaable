from diffaaable.vectorial import vectorial_aaa
import numpy as np
from baryrat import BarycentricRational
from diffaaable.core import poles

def tensor_aaa(z_k, F_k, thres_numerical_zero = 1e-13, aaa_tol=1e-9):
  total_vec = np.array([np.array(F_ki) for F_ki in F_k]).reshape(len(z_k), -1)

  norm = np.max(np.abs(total_vec), axis=0)
  numerical_zeros = norm < thres_numerical_zero
  norm = norm**0.02

  total_no_zeros = total_vec[:, ~numerical_zeros]
  norm_no_zeros = norm[~numerical_zeros]

  unique, inv_unique_idx = np.unique(total_no_zeros, axis=1, return_inverse=True)

  norm_unique = unique/norm_no_zeros #in the following we abbreiate _unique as _u

  z_j, norm_f_j_u, w_j, z_n = vectorial_aaa(z_k, norm_unique, tol=aaa_tol)
  f_j_u = norm_f_j_u * norm_no_zeros
  f_j_no_zeros = f_j_u[:, inv_unique_idx]

  f_j_vec = np.zeros((len(z_j), *total_vec.shape[1:]), dtype=complex)

  f_j_vec[:, ~numerical_zeros] = f_j_no_zeros

  f_j = f_j_vec.reshape((-1, *F_k.shape[1:]))

  # print(f"{f_j_u.shape=}")
  # print(f"{f_j_no_zeros.shape=}")
  # print(f"{f_j_vec.shape=}")
  # print(f"{f_j.shape=}")

  z_n = poles(z_j, w_j)
  return z_j, f_j, w_j, z_n

def tensor_baryrat(z_j, f_j, w_j): #TODO write down properly
    shape = f_j[0].shape
    rs = [BarycentricRational(z_j, f_j_i, w_j) for f_j_i in f_j.reshape((len(z_j), -1)).T]

    def inner(z):
      z = np.array(z)
      res = np.array([r(z) for r in rs])
      return res.reshape((*z.shape, *shape))

    return inner
