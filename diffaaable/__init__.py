"""diffaaable - JAX-differentiable AAA algorithm"""

__version__ = "1.1.1"


from diffaaable.core import aaa
from diffaaable.adaptive import adaptive_aaa
from diffaaable.lorentz import lorentz_aaa
from diffaaable.selective import selective_subdivision_aaa
from diffaaable.set_aaa import set_aaa
from diffaaable.tensor import tensor_aaa
from diffaaable.vectorial import vectorial_aaa
from diffaaable.util import poles, residues

__all__ = ["aaa", "adaptive_aaa", "lorentz_aaa", "selective_subdivision_aaa", "set_aaa", "tensor_aaa", "vectorial_aaa", "poles", "residues"]
