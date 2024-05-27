from baryrat import BarycentricRational
import jax.numpy as np
from diffaaable.lorentz import lorentz_aaa


def f_test(z, res, poles):
  f = np.zeros_like(z)
  for r, p in zip(res, poles):
    f += 1j*r/(z-p) + 1j*np.conj(r)/(z+np.conj(p))
  return f

z_k = np.linspace(-3, 5, 100) + 0.05j

def test_lorentz_aaa():
  poles = np.array([2+0.1j, 2.7+0.3j])
  residues = np.array([1j, 2])
  z_j, f_j, w_j, errors = lorentz_aaa(z_k, f_test(z_k, residues, poles), mmax=10)

  print(f"{errors=}")

  z_j = np.concatenate([z_j, -np.conj(z_j)])
  f_j = f_j[:, 0]
  f_j = np.concatenate([f_j, np.conj(f_j)])
  w_j = np.concatenate([w_j, np.conj(w_j)])

  r = BarycentricRational(z_j, f_j, w_j)

  pol, res = r.polres()

  for pole, residue in zip(poles, residues):
    assert np.any(np.isclose(pol, pole))
    assert np.any(np.isclose(pol, -np.conj(pole)))

    assert np.any(np.isclose(res, 1j*residue))
    assert np.any(np.isclose(res, 1j*np.conj(residue)))

  print(f"poles: {pol}, residues: {res}")

if __name__ == "__main__":
  test_lorentz_aaa()
