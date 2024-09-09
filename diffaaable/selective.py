import os
import pathlib
import jax.numpy as np
import numpy as onp
from jax.tree_util import Partial
from diffaaable.core import aaa, residues
from diffaaable.adaptive import Domain, domain_mask, adaptive_aaa, next_samples_heat
import matplotlib.pyplot as plt

def reduced_domain(domain, reduction=1-1/12):
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
  domain = reduced_domain(domain)
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
    1j*domain[0].imag + z_k_r,
    1j*domain[1].imag + z_k_r,
    domain[0].real + z_k_i,
    domain[1].real + z_k_i
  ]).flatten()

def anti_domain(domain: Domain):
  return (
    domain[0].real + 1j*domain[1].imag,
    domain[1].real + 1j*domain[0].imag
    )

def domain_center(domain: Domain):
  return np.mean(np.array(domain))

def subdomains(domain: Domain, divide_horizontal: bool, center: complex=None):
  if center is None:
    center = domain_center(domain)
  left_up =    domain[0].real + 1j*domain[1].imag
  right_down = domain[1].real + 1j*domain[0].imag

  subs = [
    (center, domain[1]),
    anti_domain((left_up, center)),
    (domain[0], center),
    anti_domain((center, right_down)),
  ]

  if divide_horizontal:
    return [(subs[1][0], subs[0][1]), (subs[2][0], subs[3][1])]
  return   [(subs[2][0], subs[1][1]), (subs[3][0], subs[0][1])]


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
  #return True

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
                Dmax=30,
                use_adaptive: bool = True,
                z_k = None, f_k = None,
                divide_horizontal=True,
                debug_name = "d",
                stop = 0.1,
                batchsize=10
                ):
  """
  TODO: allow access to samples slightly outside of domain
  """

  print(f"start domain '{debug_name}', {Dmax=}")
  folder = f"debug_out/{debug_name:0<33}"
  domain_size = np.abs(domain[1]-domain[0])/2
  size = domain_size/2 # for plotting
  #plot_rect = plot_domain(domain, size=30)
  #color = plot_rect[0].get_color()

  if cutoff is None:
    cutoff = np.inf

  eval_count = 0
  if use_adaptive:
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    sampling = Partial(next_samples_heat, debug=folder,
                       stop=stop, resolution=(101, 101), batchsize=batchsize)
    if z_k is None:
      z_k = np.empty((0,), dtype=complex)
      f_k = z_k.copy()

    if len(z_k) < N/4:
      z_k_new = sample_domain(domain, N)
      f_k = np.append(f_k, f(z_k_new))
      z_k = np.append(z_k, z_k_new)

      eval_count += len(z_k_new)
      print(f"new eval: {eval_count}")
    eval_count -= len(z_k)
    z_j, f_j, w_j, z_n, z_k, f_k = adaptive_aaa(
      z_k, f, f_k_0=f_k, evolutions=N*16, tol=tol_aaa,
      domain=reduced_domain(domain, 1.07), radius=4*domain_size/(N), #NOTE: actually increased domain :/
      return_samples=True, sampling=sampling, cutoff=np.inf
    )
    # TODO pass down samples in buffer zone
    eval_count += len(z_k)
  else:
    if on_rim:
      z_k = sample_rim(domain, N)
    else:
      z_k = sample_domain(reduced_domain(domain, 1.05), N)
    f_k = f(z_k)
    eval_count += len(f_k)
    try:
      z_j, f_j, w_j, z_n = aaa(z_k, f_k, tol=tol_aaa)
    except onp.linalg.LinAlgError as e:
      z_n = z_j = f_j = w_j = np.empty((0,))

  print(f"domain '{debug_name}' done: {domain} ->  eval: {eval_count}")
  poles = z_n[domain_mask(domain, z_n)]

  if (Dmax == 0 or
    (len(poles)<=max_poles and all_poles_known(poles, suggestions, tol_pol))):

    #plt.scatter(poles.real, poles.imag, color = color, marker="x")#, s=size*3, linewidths=size/2)
    print("I am done here")

    res = residues(z_j, f_j, w_j, poles)
    return poles, res, eval_count
  #plt.scatter(poles.real, poles.imag, color = color, marker="+", s=0.2, zorder=3)#, s=size, linewidths=size/6)

  subs = subdomains(domain, divide_horizontal)

  pol = np.empty((0,), dtype=complex)
  res = pol.copy()
  for i,sub in enumerate(subs):
    sug = poles[domain_mask(sub, poles)]
    sample_mask = domain_mask(sub, z_k)

    known_z_k = z_k[sample_mask]
    known_f_k = f_k[sample_mask]

    p, r, e = selective_refinement_aaa(
      f, sub, N, max_poles, cutoff, tol_aaa, tol_pol,
      use_adaptive=use_adaptive,
      suggestions=sug, Dmax=Dmax-1, z_k=known_z_k, f_k=known_f_k,
      divide_horizontal = not divide_horizontal,
      debug_name=f"{debug_name}{i+1}",
    )
    pol = np.append(pol, p)
    res = np.append(res, r)
    eval_count += e
  # if len(pol) > 0:
  #   plt.xlim(domain[0].real, domain[1].real)
  #   plt.ylim(domain[0].imag, domain[1].imag)
  #   plt.savefig(f"debug_out/{debug_name:0<33}.png")
  return pol, res, eval_count
