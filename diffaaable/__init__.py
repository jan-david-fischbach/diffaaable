"""diffaaable - JAX-differentiable AAA algorithm"""

__version__ = "0.1.0"

__all__ = ["aaa", "residues", "BarycentricRational"]

from diffaaable.diffaaable import aaa, residues
from diffaaable.adaptive import adaptive_aaa
from baryrat import BarycentricRational
