import jax.numpy as np
from diffaaable import aaa, residues
from diffaaable.adaptive import Domain, domain_mask
import matplotlib.pyplot as plt

def sample_domain(domain: Domain, N: int):
  sqrt_N = np.round(np.sqrt(N)).astype(int)
  z_k_r = np.linspace(domain[0].real, domain[1].real, sqrt_N)
  z_k_i = np.linspace(domain[0].imag, domain[1].imag, sqrt_N)
  Z_r, Z_i = np.meshgrid(z_k_r, z_k_i)
  z_k = (Z_r+1j*Z_i).flatten()
  return z_k

def anti_domain(domain: Domain):
  return (
    domain[0].real + 1j*domain[1].imag,
    domain[1].real + 1j*domain[0].imag
    )

def domain_center(domain: Domain):
  return np.mean(np.array(domain))

def subdomains(domain: Domain, center: complex=None):
  if center is None:
    center = domain_center(domain)
  left_up =    domain[0].real + 1j*domain[1].imag
  right_down = domain[1].real + 1j*domain[0].imag
  return [
    (center, domain[1]),
    anti_domain((left_up, center)),
    (domain[0], center),
    anti_domain((center, right_down)),
  ]

def cutoff_mask(z_k, f_k, f_k_dot, cutoff):
  m = np.abs(f_k)<cutoff #filter out values, that have diverged too strongly
  return z_k[m], f_k[m], f_k_dot[m]

def plot_domain(domain: Domain):
  left_up =    domain[0].real + 1j*domain[1].imag
  right_down = domain[1].real + 1j*domain[0].imag

  points = np.array([domain[0], right_down, domain[1], left_up, domain[0]])

  return plt.plot(points.real, points.imag)

def all_poles_known(poles, prev, tol):
  if len(poles) == 0:
    return True
  if prev is None or len(prev)<len(poles):
    return False
  dist = np.abs(poles[:, None] - prev[None, :])
  return np.all(np.any(dist < tol, axis=-1))

def selective_refinement_aaa(f: callable,
                domain: Domain,
                N: int = 25,
                max_poles: int = 3,
                cutoff: float = None,
                tol_aaa: float = 1e-9,
                tol_pol: float = 1e-4,
                suggestions = None):
  """
  """
  plot_rect = plot_domain(domain)

  if cutoff is None:
    cutoff = np.inf

  z_k = sample_domain(domain, N)
  f_k = f(z_k) # should probably try to reuse...
  z_j, f_j, w_j, z_n = aaa(z_k, f_k, tol=tol_aaa)

  poles = z_n[domain_mask(domain, z_n)]
  if len(poles)<=max_poles and all_poles_known(poles, suggestions, tol_pol):

    plt.scatter(poles.real, poles.imag, color = plot_rect[0].get_color())

    res = residues(z_j, f_j, w_j, poles)
    return poles, res

  center = domain_center(domain)
  subs = subdomains(domain, np.mean(np.array([np.mean(poles), center])))

  pol = np.empty((0,), dtype=complex)
  res = pol.copy()
  for sub in subs:
    p, r = selective_refinement_aaa(
      f, sub, N, max_poles, cutoff, tol_aaa, tol_pol, suggestions=poles
    )
    pol = np.append(pol, p)
    res = np.append(res, r)
  return pol, res
