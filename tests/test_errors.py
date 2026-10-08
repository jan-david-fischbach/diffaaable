import warnings
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import baryrat
from diffaaable import (
  aaa, vectorial_aaa, tensor_aaa, set_aaa, adaptive_aaa,
  InputError, ConvergenceError, ConvergenceWarning
)

z_k = np.linspace(-1, 1, 20) + 0.1j
f_k = 1/(z_k-0.3j)

variants = [
  aaa,
  lambda z, f: vectorial_aaa(z, f),
  lambda z, f: tensor_aaa(z, f[:, None]),
  lambda z, f: set_aaa(z, f[:, None]),
]

@pytest.mark.parametrize("fit", variants)
@pytest.mark.parametrize("same_value", [True, False])
def test_duplicate_samples(fit, same_value):
  # duplicate of a point that becomes a support point
  z = np.append(z_k, z_k[9])
  f = np.append(f_k, f_k[9] + (0 if same_value else 1))
  with pytest.raises(InputError, match="more than once"):
    fit(z, f)

@pytest.mark.parametrize("fit", variants)
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_samples(fit, bad):
  f = f_k.copy()
  f[5] = bad
  with pytest.raises(InputError, match="not finite"):
    fit(z_k, f)

@pytest.mark.parametrize("fit", variants)
def test_non_finite_sample_points(fit):
  z = z_k.copy()
  z[3] = np.nan
  with pytest.raises(InputError, match="not finite"):
    fit(z, f_k)

def test_shape_mismatch():
  with pytest.raises(InputError, match="same shape"):
    aaa(z_k, f_k[:-1])
  with pytest.raises(InputError, match="first dimension"):
    vectorial_aaa(z_k, f_k[:-1])
  with pytest.raises(InputError, match="first dimension"):
    tensor_aaa(z_k, f_k[:-1, None])

@pytest.mark.parametrize("fit", variants)
def test_empty(fit):
  with pytest.raises(InputError, match="No samples"):
    fit(z_k[:0], f_k[:0])

def test_input_error_is_value_error():
  with pytest.raises(ValueError):
    aaa(z_k, f_k[:-1])

def test_tensor_all_zero():
  with pytest.raises(InputError, match="numerically zero"):
    tensor_aaa(z_k, np.zeros((len(z_k), 2)))

def test_grad_duplicates():
  z = jnp.append(z_k, z_k[3])
  def loss(a):
    return jnp.real(aaa(z, 1/(z-a))[3][0])
  with pytest.raises(InputError, match="more than once"):
    jax.grad(loss)(0.3j)

def test_svd_failure(monkeypatch):
  def fail(*args, **kwargs):
    raise np.linalg.LinAlgError("SVD did not converge")
  monkeypatch.setattr(baryrat.np.linalg, "svd", fail)
  with pytest.raises(ConvergenceError, match="SVD did not converge") as e:
    aaa(z_k, f_k)
  # existing `except LinAlgError` keeps working
  assert isinstance(e.value, np.linalg.LinAlgError)

def test_mmax_warning():
  f = np.tan(5*z_k)
  with pytest.warns(ConvergenceWarning, match="mmax=2"):
    aaa(z_k, f, mmax=2)
  with pytest.warns(ConvergenceWarning, match="mmax=2"):
    vectorial_aaa(z_k, f, mmax=2)
  with pytest.warns(ConvergenceWarning, match="mmax=2"):
    set_aaa(z_k, f[:, None], mmax=2)

def test_no_warning_when_converged():
  with warnings.catch_warnings():
    warnings.simplefilter("error", ConvergenceWarning)
    aaa(z_k, f_k)
    vectorial_aaa(z_k, f_k)
    set_aaa(z_k, f_k[:, None])

def test_adaptive_closure_under_grad():
  def loss(a):
    return jnp.real(adaptive_aaa(z_k, lambda x: 1/(x-a))[3][0])
  with pytest.raises(InputError, match="Partial"):
    jax.grad(loss)(0.3j)

def test_adaptive_plain_function():
  z_n = adaptive_aaa(z_k, lambda x: 1/(x-0.3j))[3]
  assert np.min(np.abs(z_n-0.3j)) < 1e-6

def test_adaptive_all_discarded():
  with pytest.raises(InputError, match="finite"):
    adaptive_aaa(z_k, lambda x: np.full_like(x, np.nan))

def test_adaptive_cutoff_discards_all():
  with pytest.raises(InputError, match="cutoff"):
    adaptive_aaa(z_k, lambda x: 1/(x-0.3j), cutoff=1e-3)

def test_adaptive_wrong_output_shape():
  with pytest.raises(InputError, match="one sample per sample point"):
    adaptive_aaa(z_k, lambda x: x[:-1])

def test_adaptive_invalid_domain():
  with pytest.raises(InputError, match="lower_left"):
    adaptive_aaa(z_k, lambda x: 1/(x-0.3j), domain=(1+1j, -1-1j))

def test_adaptive_evolutions():
  with pytest.raises(InputError, match="evolutions"):
    adaptive_aaa(z_k, lambda x: 1/(x-0.3j), evolutions=0)
