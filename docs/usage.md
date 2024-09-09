---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# Usage

Just use it as an (almost) drop-in replacement for `baryrat`. Note how the nodes, values and weights of the barycentric approximation are returned instead of a callable barycentric rational.

```{code-cell}
:tags: [remove-stderr]
from diffaaable import aaa
import jax
import jax.numpy as jnp

### sample points ###
z_k_r = z_k_i = jnp.linspace(0, 3, 20)
Z_k_r, Z_k_i = jnp.meshgrid(z_k_r, z_k_r)
z_k = Z_k_r + 1j*Z_k_i

### function to be approximated ###
def f(x, a):
    return jnp.tan(a*x)
f_pi = jax.tree_util.Partial(f, jnp.pi)

### alternatively use pre-calculated function values ###
z_j, f_j, w_j, z_n = aaa(z_k, f_pi(z_k))

z_n
```

## Gradients

`diffaaable` is JAX differentiable. Thus you can freely compose it with other JAX functionality and obtain gradients.

```{code-cell}
def loss(a):
    f_k = f(z_k, a)

    z_j, f_j, w_j, z_n = aaa(z_k, f_k)

    selected_poles = z_n[z_n.real>1e-2]
    relevant_pole = selected_poles[jnp.argmin(selected_poles.real)]
    return jnp.real(relevant_pole - 2)

g = jax.grad(loss)
g(jnp.pi/2)
```
