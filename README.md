# diffaaable 0.1.0

![Schematic](docs/assets/diffaaable.png)

A JAX differentiable version of the AAA algorithm. The derivatives are implemented as custom Jacobian Vector products in accordance to [1].
A detailed derivation of the used matrix expressions is provided in the appendix of [2].
Under the hood `diffaaable` uses the AAA implementation in [`baryrat`](https://github.com/c-f-h/baryrat).

## Usage

Just use it as an (almost) drop-in replacement for `baryrat`. Note how the nodes, values and weights of the barycentric approximation are returned instead of a callable function.

```python
from diffaaable import aaa

z_j, f_j, w_j, z_n = aaa(z_k, f_k)
```

## Installation
to install `diffaaable` simply run
`pip install diffaaable`

## Citation
When using this software package for scientific work please cite the associated publication [2].


