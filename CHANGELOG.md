# CHANGELOG

<!-- towncrier release notes start -->

## [1.5.0](https://github.com/jan-david-fischbach/diffaaable/releases/tag/v1.5.0) - 2026-10-08


### Added

- Descriptive errors: all AAA variants validate their samples and raise an `InputError` (a `ValueError`) for duplicate or non-finite sample points, non-finite samples, mismatched shapes and empty inputs. Failing linear algebra raises a `ConvergenceError` (a `LinAlgError`) explaining likely causes, and a `ConvergenceWarning` is emitted when `tol` is not reached within `mmax` iterations. `adaptive_aaa` explains invalid domains, samples discarded by `cutoff` and functions closing over differentiated values.
- Documentation example reproducing the T-matrix pole-expansion of a chiral core-shell sphere, and a reference to the associated ACS Photonics paper.
- `tensor_baryrat` is now part of the public API: a JAX-differentiable evaluator for barycentric rationals with scalar, vector or tensor values (previously a loop over `baryrat` instances).


### Changed

- `lorentz_aaa` now returns the poles like the other variants: `z_j, f_j, w_j, z_n` (plus `errors` with `return_errors=True`).


### Fixed

- `aaa` no longer silently accepts `f_k` with a different length than `z_k`.

## [1.4.0](https://github.com/jan-david-fischbach/diffaaable/releases/tag/v1.4.0) - 2026-10-02


### Added

- JAX derivatives (custom JVPs) for `vectorial_aaa`, `set_aaa` and `tensor_aaa`.


### Changed

- `vectorial_aaa` returns numpy arrays; adaptive refinement draws its random offsets with numpy (sample positions differ from earlier versions).


### Performance

- `adaptive_aaa` and `selective_subdivision_aaa` up to 20x faster by avoiding JAX recompilation in host-side bookkeeping.
- `set_aaa` and `tensor_aaa` up to 70x faster for small and 2x faster for large problems.
- `vectorial_aaa` over 100x faster (numpy with reduced SVD).


### Fixed

- `adaptive_aaa` gradients differentiate the selected `aaa` variant with the given `tol`/`mmax` and work with plain functions as `aaa`/`sampling` and with `return_samples=True`.


### Removed

- Dropped support for Python 3.9 (end of life since October 2025); diffaaable now requires Python >= 3.10.


### Security

- Updated `uv.lock` to patched versions of pillow, tornado, urllib3, requests, soupsieve, filelock, idna, pytest, setuptools and pygments, resolving all open Dependabot alerts (dev/docs dependencies only).
- Updated locked `virtualenv` (a pre-commit dependency) from 21.2.0 to 21.14.3, resolving four Dependabot alerts (activation-script command injection, unverified seed wheels, `pyvenv.cfg` injection).

## [1.3.1](https://github.com/jan-david-fischbach/diffaaable/releases/tag/v1.3.1) - 2026-03-04

No significant changes.


## [1.3.0](https://github.com/jan-david-fischbach/diffaaable/releases/tag/v1.3.0) - 2026-03-04

No significant changes.


## [1.2.1](https://github.com/jan-david-fischbach/diffaaable/releases/tag/v1.2.1) - 2026-01-12

No significant changes.


## [1.2.0](https://github.com/jan-david-fischbach/diffaaable/releases/tag/v1.2.0) - 2025-05-06

No significant changes.


## [1.1.1](https://github.com/jan-david-fischbach/diffaaable/releases/tag/v1.1.1) - 2025-02-10

No significant changes.


## [1.1.0](https://github.com/jan-david-fischbach/diffaaable/releases/tag/v1.1.0) - 2025-01-31

No significant changes.


## [1.0.1](https://github.com/jan-david-fischbach/diffaaable/releases/tag/v1.0.1) - 2024-09-11

No significant changes.


## [1.0.0](https://github.com/jan-david-fischbach/diffaaable/releases/tag/v1.0.0) - 2024-09-11

No significant changes.


## [0.1.0](https://gitlab.kit.edu/kit/tfp-photonics/solar/diffaaable/releases/tag/v0.1.0) - 2024-02-15

No significant changes.
