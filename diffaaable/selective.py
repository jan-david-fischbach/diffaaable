import os
import jax.numpy as np
from jax.tree_util import Partial
from diffaaable import aaa, residues
from diffaaable.adaptive import Domain, domain_mask, adaptive_aaa, next_samples_heat
import matplotlib.pyplot as plt

def increased_domain(domain, reduction=1+1e-2):
  r = reduction
  return (
    domain[0]*r+domain[1]*(1-r),
    domain[1]*r+domain[0]*(1-r)
  )

def sample_cross(domain):
  center = domain_center(domain)
  dist = 0.5 * (domain[1]-domain[0])
  return center+np.array([dist.real, -dist.real, 1j*dist.imag, -1j*dist.imag])

def sample_domain(domain: Domain, N: int):
  sqrt_N = np.round(np.sqrt(N)).astype(int)
  domain = increased_domain(domain)
  z_k_r = np.linspace(domain[0].real, domain[1].real, sqrt_N)
  z_k_i = np.linspace(domain[0].imag, domain[1].imag, sqrt_N)
  Z_r, Z_i = np.meshgrid(z_k_r, z_k_i)
  z_k = (Z_r+1j*Z_i).flatten()
  return z_k

def sample_rim(domain: Domain, N: int):
  side_N = N//4
  z_k_r = np.linspace(domain[0].real, domain[1].real, side_N+2)[1:-1]
  z_k_i = np.linspace(domain[0].imag, domain[1].imag, side_N+2)[1:-1] * 1j
  return np.array([
    domain[0].imag + z_k_r,
    domain[1].imag + z_k_r,
    domain[0].real + z_k_i,
    domain[1].real + z_k_i
  ])

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

def plot_domain(domain: Domain, size: float=1):
  left_up =    domain[0].real + 1j*domain[1].imag
  right_down = domain[1].real + 1j*domain[0].imag

  points = np.array([domain[0], right_down, domain[1], left_up, domain[0]])

  return plt.plot(points.real, points.imag,
                  lw=size/30, zorder=1)

def all_poles_known(poles, prev, tol):
  if prev is None or len(prev)!=len(poles):
    return False
  return True

  dist = np.abs(poles[:, None] - prev[None, :])
  check = np.all(np.any(dist < tol, axis=1))
  return check


def selective_refinement_aaa(f: callable,
                domain: Domain,
                N: int = 36,
                max_poles: int = 400,
                cutoff: float = None,
                tol_aaa: float = 1e-9,
                tol_pol: float = 1e-5,
                suggestions = None,
                on_rim: bool = False,
                Dmax=20,
                use_adaptive: bool = True,
                z_k = None, f_k = None,
                debug_name = "d"
                ):
  """
  """
  if Dmax == 0:
    return np.array([]), np.array([]), 0

  domain_size = np.abs(domain[1]-domain[0])/2
  size = domain_size*1 # for plotting
  plot_rect = plot_domain(domain, size=size)
  color = plot_rect[0].get_color()

  if cutoff is None:
    cutoff = np.inf

  eval_count = 0
  if use_adaptive:
    folder = f"debug_out/{debug_name}"
    os.mkdir(folder)
    sampling = Partial(next_samples_heat, debug=folder,
                       stop=0.2)
    if z_k is None:
      z_k = sample_domain(domain, N)
      f_k = f(z_k)
      eval_count += len(f_k)
      print(f"init eval: {eval_count}")
    eval_count -= len(z_k)
    z_j, f_j, w_j, z_n, z_k, f_k = adaptive_aaa(
      z_k, f, f_k_0=f_k, evolutions=N, tol=tol_aaa,
      domain=domain, radius=domain_size/N,
      return_samples=True, sampling=sampling, cutoff=np.inf
    )
    eval_count += len(z_k)
  else:
    if on_rim:
      z_k = sample_rim(domain, N)
    else:
      z_k = sample_domain(domain, N)
    f_k = f(z_k)
    eval_count += len(f_k)

    z_j, f_j, w_j, z_n = aaa(z_k, f_k, tol=tol_aaa)

  print(f"domain '{debug_name}': {domain} ->  eval: {eval_count}")
  poles = z_n[domain_mask(domain, z_n)]

  if len(poles)<=max_poles and all_poles_known(poles, suggestions, tol_pol):
    plt.scatter(poles.real, poles.imag, color = color, marker="x", s=size*3, linewidths=size/2)
    plt.savefig("debug_out/selective.png")

    res = residues(z_j, f_j, w_j, poles)
    return poles, res, eval_count
  plt.scatter(poles.real, poles.imag, color = color, marker="+", s=size, linewidths=size/6, zorder=3)

  subs = subdomains(domain)

  pol = np.empty((0,), dtype=complex)
  res = pol.copy()
  for i,sub in enumerate(subs):
    sug = poles[domain_mask(sub, poles)]
    sample_mask = domain_mask(sub, z_k)
    if np.sum(sample_mask) > 2:
      known_z_k = z_k[sample_mask]
      known_f_k = f_k[sample_mask]
    else:
      known_z_k = None
      known_f_k = None

    p, r, e = selective_refinement_aaa(
      f, sub, N, max_poles, cutoff, tol_aaa, tol_pol,
      suggestions=sug, Dmax=Dmax-1, z_k=known_z_k, f_k=known_f_k,
      debug_name=f"{debug_name}{i}"
    )
    pol = np.append(pol, p)
    res = np.append(res, r)
    eval_count += e
  return pol, res, eval_count
