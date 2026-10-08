import scipy.linalg
import numpy as np
import jax
import jax.numpy as jnp
from diffaaable.errors import InputError

def _describe(values, max_shown=5):
  values = np.asarray(values)
  shown = ", ".join(str(v) for v in values[:max_shown])
  if len(values) > max_shown:
    shown += f", ... ({len(values)} in total)"
  return shown

def validate_samples(z_k, f_k, vector_valued=False):
  """
  Check the samples passed to an AAA variant and raise an `InputError`
  describing the problem if they are invalid.

  Parameters
  ----------
    z_k : array (M,)
      sample points. Without `vector_valued` any shape is accepted (and
      flattened) as long as it matches `f_k`.
    f_k : array (M, ...)
      samples
    vector_valued : bool
      whether `f_k` holds (M, ...) vector/tensor valued samples

  Raises
  ------
    InputError
      if the shapes do not match, no samples are provided, any sample point
      or sample is not finite, or a sample point occurs more than once.
  """
  z_shape, f_shape = np.shape(z_k), np.shape(f_k)
  if vector_valued:
    if len(z_shape) != 1:
      raise InputError(f"z_k has to be 1D but has shape {z_shape}")
    if len(f_shape) < 1 or f_shape[0] != z_shape[0]:
      raise InputError(
        "The first dimension of f_k has to match the number of sample points "
        f"z_k, but f_k.shape={f_shape} and z_k.shape={z_shape}"
      )
  elif z_shape != f_shape:
    raise InputError(
      "z_k and f_k need to have the same shape (one sample per sample point), "
      f"but z_k.shape={z_shape} and f_k.shape={f_shape}"
    )

  M = int(np.prod(z_shape[:1] if vector_valued else z_shape))
  if M == 0:
    raise InputError("No samples provided: z_k and f_k are empty")

  # values of tracers are unknown (e.g. under jax.jit) -> skip value checks
  if not isinstance(z_k, jax.core.Tracer):
    z = np.ravel(z_k)
    bad = np.flatnonzero(~np.isfinite(z))
    if len(bad):
      raise InputError(
        f"{len(bad)} of {M} sample points z_k are not finite (nan or inf) "
        f"at indices {_describe(bad)}"
      )

    unique, counts = np.unique(z, return_counts=True)
    duplicates = unique[counts > 1]
    if len(duplicates):
      raise InputError(
        f"{len(duplicates)} sample point(s) occur more than once in z_k: "
        f"{_describe(duplicates)}. AAA requires distinct sample points; "
        "remove the duplicates, e.g. using `np.unique(z_k, return_index=True)`."
      )

  if not isinstance(f_k, jax.core.Tracer):
    f = np.reshape(np.asarray(f_k), (M, -1))
    bad = np.flatnonzero(~np.all(np.isfinite(f), axis=1))
    if len(bad):
      raise InputError(
        f"{len(bad)} of {M} samples f_k are not finite (nan or inf) at "
        f"indices {_describe(bad)} (sample points z_k: "
        f"{_describe(np.ravel(z_k)[bad])}). Discard these samples, e.g. "
        "using the `cutoff` argument of `adaptive_aaa`."
      )

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

def node_indices(z_k, z_j):
  """
  Indices of the support points `z_j` within the samples `z_k`.

  Parameters
  ----------
    z_k : array (M,)
      sample points
    z_j : array (m,)
      support points (a subset of `z_k`)

  Returns
  -------
    idx : array (m,)
      `z_k[idx] == z_j` (first occurrence)
  """
  z_k = np.asarray(z_k)
  z_j = np.asarray(z_j)
  match = z_j[:, None] == z_k[None, :]
  if not np.all(np.any(match, axis=1)):
    raise ValueError("The support points z_j have to be a subset of the samples z_k")
  return np.argmax(match, axis=1)

def weights_jvp(z_k, f_k, f_dot, z_j, w_j, entry_weights=None):
  r"""
  Tangents of the barycentric weights $w_j$ given the tangents of the
  samples $f_k$ (see `diffaaable.core.aaa_jvp` for the derivation).

  Vector/tensor valued samples share the weights. Their equations are stacked
  into a single least squares problem.

  Parameters
  ----------
    z_k : array (M,)
      sample points
    f_k : array (M, ...)
      samples
    f_dot : array (M, ...)
      tangents of the samples
    z_j : array (m,)
      support points (a subset of `z_k`)
    w_j : array (m,)
      barycentric weights
    entry_weights : array (V,), optional
      scaling of the (flattened) entries in the least squares problem,
      mirroring a normalization applied when fitting. Entries with zero
      weight are excluded.

  Returns
  -------
    w_j_dot : array (m,)
  """
  z_k = np.asarray(z_k)
  M = len(z_k)
  f_k = jnp.reshape(jnp.asarray(f_k), (M, -1))
  f_dot = jnp.reshape(f_dot, (M, -1))

  if entry_weights is not None:
    entry_weights = np.asarray(entry_weights).reshape(-1)
    keep = (entry_weights != 0) & np.isfinite(entry_weights)
    f_k = f_k[:, keep] * entry_weights[keep]
    f_dot = f_dot[:, keep] * entry_weights[keep]

  idx = node_indices(z_k, z_j)
  f_j = f_k[idx]
  f_j_dot = f_dot[idx]

  rest = ~np.isin(z_k, z_j)
  z_k, f_k, f_dot = z_k[rest], f_k[rest], f_dot[rest]

  C = 1/(z_k[:, None]-z_j[None, :]) # Cauchy matrix k x j

  d = C @ w_j # denominator in barycentric formula
  via_f_j = C @ (f_j_dot * w_j[:, None]) / d[:, None] # $\sum_j f_j^\prime \frac{\del r}{\del f_j}$

  A = (f_j[None, :, :] - f_k[:, None, :])*(C/d[:, None])[:, :, None] # k x j x V
  b = f_dot - via_f_j # k x V

  # stack the equations of all entries
  A = jnp.moveaxis(A, -1, 0).reshape(-1, len(z_j))
  b = b.T.reshape(-1)

  # make sure system is not underdetermined according to eq. 5 of [1]
  A = jnp.concatenate([A, jnp.conj(w_j.reshape(1, -1))])
  b = jnp.append(b, 0)

  with jax.disable_jit(): #otherwise backwards differentiation led to error
    w_j_dot, _, _, _ = jnp.linalg.lstsq(A, b)
  return w_j_dot

def poles_jvp(z_j, w_j, z_n, w_j_dot):
  """
  Tangents of the poles `z_n` given the tangents of the weights `w_j`.
  """
  denom = z_n.reshape(1, -1)-z_j.reshape(-1, 1)
  return (
    jnp.sum(w_j_dot.reshape(-1, 1)/denom,    axis=0)/
    jnp.sum(w_j.reshape(-1, 1)    /denom**2, axis=0)
  )

def check_z_dot(z_dot):
  """Raise if the sample points `z_k` carry a (non zero) tangent."""
  if np.any(z_dot):
    raise NotImplementedError(
      "Differentiating with respect to the sample points z_k is not "
      "supported. Only the samples f_k may depend on the parameters."
    )

def barycentric_jvp(z_k, f_k, z_dot, f_dot, z_j, f_j, w_j, z_n, entry_weights=None):
  """
  Tangents of the outputs `(z_j, f_j, w_j, z_n)` of an AAA fit to the samples
  `f_k` at `z_k`. Works for scalar (M,) as well as vector/tensor (M, ...)
  valued samples. See `weights_jvp`.
  """
  check_z_dot(z_dot)

  idx = node_indices(z_k, z_j)
  z_j_dot = z_dot[idx]
  f_j_dot = jnp.reshape(f_dot[idx], np.shape(f_j)).astype(np.result_type(f_j))

  w_j_dot = weights_jvp(z_k, f_k, f_dot, z_j, w_j, entry_weights)
  z_n_dot = poles_jvp(z_j, w_j, z_n, w_j_dot)

  return z_j_dot, f_j_dot, w_j_dot, z_n_dot

def aaa_jvp_rule(fun, entry_weights=None):
  """
  Build a `jax.custom_jvp` rule for an AAA variant
  `fun(z_k, f_k, ...) -> (z_j, f_j, w_j, z_n)`.

  Parameters
  ----------
    fun : callable
      the (non differentiable) AAA implementation
    entry_weights : callable, optional
      called with the primals, returns the `entry_weights` of `weights_jvp`
  """
  def rule(primals, tangents):
    z_k, f_k = primals[:2]
    z_dot, f_dot = tangents[:2]
    check_z_dot(z_dot)

    primal_out = fun(*primals)
    z_j, f_j, w_j, z_n = primal_out
    ew = None if entry_weights is None else entry_weights(*primals)

    tangent_out = barycentric_jvp(
      z_k, f_k, z_dot, f_dot, z_j, f_j, w_j, z_n, entry_weights=ew
    )
    return primal_out, tangent_out
  return rule
