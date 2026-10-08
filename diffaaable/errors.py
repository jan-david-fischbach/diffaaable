"""Exceptions and warnings raised by diffaaable."""
import warnings
import numpy as np


class DiffaaableError(Exception):
  """Base class of all errors raised by diffaaable."""


class InputError(DiffaaableError, ValueError):
  """The provided samples or arguments are invalid."""


class ConvergenceError(DiffaaableError, np.linalg.LinAlgError):
  """The (linear algebra of the) AAA iteration failed."""


class ConvergenceWarning(UserWarning):
  """The requested tolerance was not reached within `mmax` iterations."""


def convergence_error(err: Exception, variant: str = "AAA"):
  """Wrap a failure of the linear algebra in a `ConvergenceError` explaining
  likely causes."""
  return ConvergenceError(
    f"The linear algebra of {variant} failed ({type(err).__name__}: {err}). "
    "The input samples passed validation (finite, unique sample points), so "
    "this is most likely caused by extreme magnitudes in `f_k`, e.g. samples "
    "very close to a pole. Consider discarding such samples (see the "
    "`cutoff` argument of `adaptive_aaa`), rescaling `f_k`, or reducing "
    "`mmax`."
  )


def mmax_warning(mmax: int, error: float, tol: float, variant: str = "AAA"):
  """Warn that the tolerance was not reached within `mmax` iterations."""
  warnings.warn(
    f"{variant} did not reach the tolerance tol={tol:g} within mmax={mmax} "
    f"iterations (final error {error:.3g}, on the scale of tol). The "
    "approximation may be inaccurate. Consider increasing `mmax`, relaxing "
    "`tol`, or checking `f_k` for noise.",
    ConvergenceWarning, stacklevel=2
  )
