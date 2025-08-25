from diffaaable.set_aaa import set_aaa
import numpy as np
from baryrat import BarycentricRational
from diffaaable.util import poles
from diffaaable.vectorial import residues_vec

def tensor_aaa(z_k, F_k, tol_aaa=1e-9, mmax_aaa=100, thres_numerical_zero = 1e-13, norm_power=0):
  """
  Convenience alternative to the vector valued AAA algorithm (`aaa.vectorial`) accepting
  a tensor valued function `F_k` (so arbitrary dimensionality) instead of the single dimension that `aaa.vectorial` requires.

  This function will first flatten the tensor valued function `F_k` and then apply the AAA algorithm to the flattened data.
  The result will be reshaped to the original tensor shape.
  The AAA algorithm will only be applied to the non-zero entries of the tensor.
  The entries very close to zero for all `z_k` will be replaced by zero.

  .. attention::
  Internally `tensor_aaa` uses a sped up version of the AAA algorithm. (see https://doi.org/10.1093/imanum/draa098)
  This can lead to numerical issues when reorthogonalization is sloppy. Further investigation needed.

  Parameters
  ----------
      z_k: complex
        M sample points
      F_k: complex
        Mx... array of the sampled vector (arbitrary size) evaluated at `z_k`
      tol_aaa: float
       tolerance for the AAA algorithm
      mmax_aaa: int
        maximum number of support points for the AAA algorithm
      thres_numerical_zero: float
        threshold for detecting numerical zeros. These will be replaced by symbolic zeros and not fitted.
      norm_power: int
        The different tensor entries are normalized by their maximum absolute value to the power of `norm_power`.
        By default `norm_power=0` the tensor entries are not normalized.
  """

  total_vec = np.array([np.array(F_ki) for F_ki in F_k]).reshape(len(z_k), -1)

  norm = np.max(np.abs(total_vec), axis=0)
  numerical_zeros = norm < thres_numerical_zero
  norm = norm**norm_power

  total_no_zeros = total_vec[:, ~numerical_zeros]
  norm_no_zeros = norm[~numerical_zeros]

  unique, unique_idx, inv_unique_idx = np.unique(
     total_no_zeros, axis=1,
     return_index=True, return_inverse=True
  )
  norm_no_zeros_unique = norm_no_zeros[unique_idx]

  norm_unique = unique/norm_no_zeros_unique #in the following we abbreiate _unique as _u

  z_j, norm_f_j_u, w_j, z_n = set_aaa(z_k, norm_unique, tol=tol_aaa, mmax=mmax_aaa, normalize=False)
  f_j_u = norm_f_j_u * norm_no_zeros_unique
  f_j_no_zeros = f_j_u[:, inv_unique_idx]

  f_j_vec = np.zeros((len(z_j), *total_vec.shape[1:]), dtype=complex)

  f_j_vec[:, ~numerical_zeros] = f_j_no_zeros

  f_j = f_j_vec.reshape((-1, *F_k.shape[1:]))

  z_n = poles(z_j, w_j)
  return z_j, f_j, w_j, z_n

def tensor_baryrat(z_j, f_j, w_j): #TODO write down properly (eg.g. using jax.vmap)
    shape = f_j[0].shape
    rs = [BarycentricRational(z_j, f_j_i, w_j) for f_j_i in f_j.reshape((len(z_j), -1)).T]

    def inner(z):
      z = np.array(z)
      results = [r(z) for r in rs]
      res = np.stack(results, axis=-1)
      out_shape = (*z.shape, *shape)
      return res.reshape(out_shape)

    return inner

def resiudes(z_j,f_j,w_j,z_n):
  return residues_vec(
      z_j, f_j.reshape((len(f_j), -1)).T, w_j, z_n
    ).T.reshape((len(z_n), f_j[0].shape))
