"""diffaaable - JAX-differentiable AAA algorithm"""

__version__ = "1.1.0"

__all__ = ["aaa", "adaptive_aaa", "vectorial_aaa", "lorentz_aaa"]

from diffaaable.core import aaa
from diffaaable.adaptive import adaptive_aaa
from diffaaable.vectorial import vectorial_aaa
from diffaaable.lorentz import lorentz_aaa
