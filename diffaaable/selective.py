import jax.numpy as np
import numpy as onp
from diffaaable.core import aaa
from diffaaable.util import residues
from diffaaable.adaptive import Domain, domain_mask, adaptive_aaa
import matplotlib.pyplot as plt

def reduced_domain(domain, reduction=1-1/12):
  """
  Utility: Rescale the domain. Can be used to shring or enlarge the domain.

  Parameters
  ----------
  domain: Domain
    The domain to rescale.
  reduction: float
    The factor by which to rescale the domain. A value of 1 will not change the domain. A value of 0.5 will shrink the domain by half. A value of 2 will enlarge the domain by a factor of 2.
  """

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
  """
  Utility: Subdivide the domain into two subdomains.

  Parameters
  ----------
  domain: Domain
    The domain to subdivide.
  divide_horizontal: bool
    If True, the domain is divided horizontally. If False, the domain is divided vertically.
  center: complex
    The center through wich to divide the domain. If None, the center is calculated as the mean of the domain.
  """
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

def plot_domain(domain: Domain, size: float=1):
  """
  Utility: Plot the domain as a rectangle.

  Parameters
  ----------
  domain: Domain
    The domain to plot.
  size: float
    The relative size to scale the linewidth of the rectangle.

  TODO
  ----
  - Add a color argument to the function.
  """
  left_up =    domain[0].real + 1j*domain[1].imag
  right_down = domain[1].real + 1j*domain[0].imag

  points = np.array([domain[0], right_down, domain[1], left_up, domain[0]])

  return plt.plot(points.real, points.imag,
                  lw=size/30, zorder=1)

def all_poles_known(poles, prev, tol):
  if prev is None or len(prev)!=len(poles):
    return False

  dist = np.abs(poles[:, None] - prev[None, :])
  check = np.all(np.any(dist < tol, axis=1))
  return check


def selective_subdivision_aaa(f: callable,
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
                evolutions_adaptive: int = 5,
                radius_adaptive: float = 1e-4,
                z_k = None, f_k = None,
                divide_horizontal=True,
                debug_plot_domains: bool = False,
                ):
  """
  When the number of poles that need to be located is large it can be beneficial to subdivide the search domain.
  This function implements a recursive subdivision of the domain, that automatically terminates once the poles are found to a satisfactory degree.

  Parameters
  ----------
  f: callable
    The function that the pole search is conducted on. It should accept batches of complex numbers and return a complex number per input.
  domain: Domain
    The pole search is limited to this domain.
  N: int
    The initial number of samples. If the number of samples drops below N the algorithm will add N new samples to the domain.
  max_poles: int
    The maximum number of poles that are considered valid within one search. If this number is exceeded the domain will be subdivided.
  cutoff: float
    A cutoff to avoid numerical instability due to large samples close to poles. See also `diffaaable.adaptive.adaptive_aaa`.
  tol_aaa: float
    The tolerance for the AAA algorithm. See also `diffaaable.core.aaa`.
  tol_pol: float
    The tolerance for the pole search. This is used to determine if a pole has moved significantly since the last domain subdivision.
  suggestions: array
    A list of poles that are already known. This is used internally to recursively call `selective_subdivision_aaa`.
  on_rim: bool
    If True, the initial samples are taken on the rim of the domain. Selecting the samples on the domain border is closely
    related to similar contour integral approaches. It is however generally recomended to sample within the domain (default: False).
  Dmax: int
    The maximum number of subdivisions. If this number is exceeded the algorithm will stop and return the current poles.
    TODO: allow the user to specify that when reaching Dmax the algorithm should return no poles to avoid false poles.
  use_adaptive: bool
    If True, the algorithm will use the adaptive AAA algorithm within the subdomains to locate the poles more accurately.
    The samples collected while searching the parent domain are passed to the respective subdomains to minimize computational cost.
    Using the adaptive aaa is generally recommended (default: True). If False, the algorithm will use the standard AAA algorithm.
  evolutions_adaptive: int
    The number of evolutions for the adaptive AAA algorithm.
  radius_adaptive: float
    The radius for the adaptive AAA algorithm. See also `diffaaable.adaptive.adaptive_aaa`.
  z_k: array
    The samples that have already been collected. This is used internally to recursively call `selective_subdivision_aaa`. It can also be used by the user to pass samples that are already known.
  f_k: array
    The function values of the samples that have already been collected. This is used internally to recursively call `selective_subdivision_aaa`. It can also be used by the user to pass samples that are already known.
  divide_horizontal: bool
    If True, the next domain division will be horizontal. Used internally to alternate between horizontal and vertical divisions during recursion.
  debug_plot_domains: bool
    If True, the algorithm will plot the domains that are being searched. This is useful for debugging and understanding the algorithm.


  TODO
  ----
  - allow access to samples slightly outside of domain
  - divide horizontal/vertical according to the distribution of poles
  """

  domain_size = np.abs(domain[1]-domain[0])/2

  if debug_plot_domains:
    print(f"Domain: {domain}")
    plot_domain(domain, size=domain_size)

  if cutoff is None:
    cutoff = np.inf

  eval_count = 0
  if use_adaptive:
    if z_k is None:
      z_k = np.empty((0,), dtype=complex)
      f_k = z_k.copy()

    if len(z_k) < N:
      z_k_new = sample_domain(domain, N)
      f_k = np.append(f_k, f(z_k_new))
      z_k = np.append(z_k, z_k_new)

      eval_count += len(z_k_new)

    eval_count -= len(z_k)

    # NOTE: reduced domain with a factor larger than 1
    # actually increases domain size to avoid missing poles right at the border
    z_j, f_j, w_j, z_n, z_k, f_k = adaptive_aaa(
      z_k, f, f_k_0=f_k, evolutions=evolutions_adaptive, tol=tol_aaa,
      domain=reduced_domain(domain, 1.07), radius=domain_size*radius_adaptive,
      return_samples=True, cutoff=cutoff
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

  poles = z_n[domain_mask(domain, z_n)]

  if (Dmax == 0 or
    (len(poles)<=max_poles and all_poles_known(poles, suggestions, tol_pol))):

    res = residues(z_j, f_j, w_j, poles)
    return poles, res, eval_count

  subs = subdomains(domain, divide_horizontal)

  pol = np.empty((0,), dtype=complex)
  res = pol.copy()
  for i,sub in enumerate(subs):
    sug = poles[domain_mask(sub, poles)]
    sample_mask = domain_mask(sub, z_k)

    known_z_k = z_k[sample_mask]
    known_f_k = f_k[sample_mask]

    p, r, e = selective_subdivision_aaa(
      f, sub, N, max_poles, cutoff, tol_aaa, tol_pol,
      use_adaptive=use_adaptive,
      evolutions_adaptive=evolutions_adaptive,
      radius_adaptive=radius_adaptive, on_rim=on_rim,
      suggestions=sug, Dmax=Dmax-1, z_k=known_z_k, f_k=known_f_k,
      divide_horizontal = not divide_horizontal,
      debug_plot_domains=debug_plot_domains
    )
    pol = np.append(pol, p)
    res = np.append(res, r)
    eval_count += e
  return pol, res, eval_count
