from jax import config
config.update("jax_enable_x64", True) #important -> else aaa fails
import jax.numpy as jnp
import numpy.typing as npt
import jax
import numpy as np
from baryrat import aaa as oaaa # ordinary aaa
import functools
import scipy.linalg

@functools.wraps(oaaa)
@jax.custom_jvp
def aaa(z_k: npt.NDArray, f_k: npt.NDArray, tol: float=1e-13, mmax: int=100):
  """
  Wraped aaa to enable JAX based autodiff.
  """
  r = oaaa(z_k, f_k, tol=tol, mmax=mmax)
  z_j = r.nodes
  f_j = r.values
  w_j = r.weights

  mask = w_j!=0
  z_j = z_j[mask]
  f_j = f_j[mask]
  w_j = w_j[mask]

  z_n = poles(z_j, w_j)

  z_n = z_n[jnp.argsort(-jnp.abs(z_n))]

  return z_j, f_j, w_j, z_n

delimiter = '    \n'

aaa.__doc__ = f"""\
  This is a wrapped version of `aaa` as provided by `baryrat`,
  providing a custom jvp to enable differentiability.

  For detailed information on the usage of `aaa` please refer to
  the original documentation::

    {delimiter.join(aaa.__doc__.splitlines())}

  .. attention::
    Returns nodes, values, weights and poles, in contrast to the
    baryrat implementation that returns the BarycentricRational ready to be
    evaluated. This is done to facilitate differentiability.

  Parameters
  ----------
    z_k : array (M,)
      the sampling points of the function. Unlike for interpolation
      algorithms, where a small number of nodes is preferred, since the
      AAA algorithm chooses its support points adaptively, it is better
      to provide a finer mesh over the support.
    f_k : array (M,)
      the function to be approximated; can be given as a callable function
      or as an array of function values over `z_k`.
    tol : float
      the approximation tolerance
    mmax : int
      the maximum number of iterations/degree of the resulting approximant

  Returns
  -------
    z_j : array (m,)
      nodes of the barycentric approximant

    f_j : array (m,)
      values of the barycentric approximant

    w_j : array (m,)
      weights of the barycentric approximant

    z_n : array (m-1,)
      poles of the barycentric approximant (for convenience)

"""

@aaa.defjvp
def aaa_jvp(primals, tangents):
  r"""Derivatives according to [1].
  The implemented matrix expressions are motivated in the appendix of [2]:

  .. topic:: AAA Derivatives

    Here we will briefly elaborate how the derivatives introduced in [1] are
    implemented as JAX compatible Jacobian Vector products (JVPs) in `diffaaable`.

    Given the tangents $\frac{\partial f_k}{\partial p}$ we will use the chain rule on $r(w_j, f_j, z)$ along its weights $w_j$ and values $f_j$ (the nodes $z_j$ are treated as independent of $p$) (Equation 4 of [1]):
    \[
    \frac{\partial f_k}{\partial p} \approx \frac{\partial r_k}{\partial p}= \sum_{j=1}^m\frac{\partial r_k}{\partial f_j}\frac{\partial f_j}{\partial p}+\sum_{j=1}^m\frac{\partial r_k}{\partial w_j}\frac{\partial w_j}{\partial p}
    \]

    To solve this system of equations for $\frac{\partial w_j}{\partial p}$ we express it in matrix form:

    .. math::
      \mathbf{A}\mathbf{w}^\prime = \mathbf{b}

    where $\mathbf{w}^\prime$ is the column vector containing $\frac{\partial w_j}{\partial p}$ and $\mathbf{b}$ and $\mathbf{A}$ are defined element wise:

    .. math::
      \begin{aligned}
          b_k &= \frac{\partial f_k}{\partial p} - \sum_{j=1}^m\frac{\partial r_k}{\partial f_j}\frac{\partial f_j}{\partial p}\\
          A_{kj} &= \frac{\partial r_k}{\partial w_j}
      \end{aligned}


    These are augmented with Equation 5 of \cite{betzEfficientRationalApproximation2024} which removes the ambiguity in $\mathbf{w}^\prime$ associated with a shared phase of all weights.

    The expressions for the derivatives of $r_k$ are found in the definition of $r(z)$ (Equation 1 in [2]): \[ \frac{\partial r_k}{\partial f_j}= \frac{1}{d(z_k)} \frac{w_j}{z_k-z_j}\] and
    \[
    \frac{\partial r_k}{\partial w_j}= \frac{1}{d(z_k)} \frac{f_j-r_k}{z_k-z_j} \approx \frac{1}{d(z_k)} \frac{f_j-f_k}{z_k-z_j}
    \]

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

  denom = z_n.reshape(1, -1)-z_j.reshape(-1, 1)
  # jax.debug.print("wj: {}", w_j.reshape(-1, 1))
  # jax.debug.print("denom^2: {}", denom**2)
  z_n_dot = (
    jnp.sum(w_j_dot.reshape(-1, 1)/denom,    axis=0)/
    jnp.sum(w_j.reshape(-1, 1)    /denom**2, axis=0)
  )

  tangent_out = z_j_dot, f_j_dot, w_j_dot, z_n_dot

  return primal_out, tangent_out

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
