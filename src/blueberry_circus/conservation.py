"""Conservation-law certificates for closed, conservative systems.

A closed central-force system integrated with ``rr="none"`` and no external
field must conserve mechanical energy ``E = 1/2 m v^2 + U(r)`` and, because the
force is central, the angular-momentum magnitude ``|L| = |m x cross v|``. These
are re-checkable :class:`~blueberry_circus.certify.Certificate` claims in exactly
the repo's assurance style: the residual is the fractional drift of the quantity
over the window, encoded against the canonical ``residual_le_tol`` rule, so a
tampered trajectory (or an under-resolved integration) re-derives ``FAIL``.

They are the correctness backbone for the central-force potential family: any new
conservative potential can self-certify that its analytic force is consistent
with its potential energy, purely from the integrated dynamics.
"""
from __future__ import annotations

import numpy as np

from .certify import Certificate, finalize
from .observables import total_energy, angular_momentum

_METHOD = "blueberry_circus closed-system RK4 conservation check"


def _fractional_drift(series) -> float:
    """(max - min) / |mean| -- the fractional peak-to-peak drift of a conserved
    quantity. Returns a finite over-threshold sentinel if the series is not all
    finite, so the certificate stays canonicalizable and re-derives FAIL."""
    series = np.asarray(series, float)
    if not np.all(np.isfinite(series)):
        return np.inf
    mean = series.mean()
    denom = abs(mean) if mean != 0.0 else 1.0
    return float((series.max() - series.min()) / denom)


def _residual_certificate(kind, claim, value, residual, tolerance, provenance):
    # Keep the hash surface clean: a non-finite drift is recorded as a finite
    # FAIL sentinel and the value is dropped to None (mirrors rel_error_certificate).
    if not np.isfinite(residual):
        residual = abs(float(tolerance)) * 2.0 + 1.0
        value = None
    cert = Certificate(
        kind=kind, claim=claim, method=_METHOD, rule="residual_le_tol",
        value=(float(value) if value is not None else None),
        residual=float(residual), tolerance=float(tolerance),
        provenance=provenance)
    return finalize(cert)


def energy_conservation_certificate(traj, potential, particle, tol=1e-6):
    """Certify that mechanical energy drifts by less than ``tol`` (fractional)."""
    E = total_energy(traj, potential, particle)
    return _residual_certificate(
        kind="energy_conservation",
        claim="mechanical energy is conserved along the trajectory",
        value=(np.mean(E) if np.all(np.isfinite(E)) else None),
        residual=_fractional_drift(E), tolerance=tol,
        provenance={"n_steps": int(len(traj.t)),
                    "quantity": "E_peak_to_peak_over_mean"})


def angular_momentum_conservation_certificate(traj, particle, tol=1e-6):
    """Certify that ``|L|`` drifts by less than ``tol`` (fractional) -- the
    signature of a central force."""
    Lmag = np.linalg.norm(angular_momentum(traj, particle), axis=1)
    return _residual_certificate(
        kind="angular_momentum_conservation",
        claim="angular-momentum magnitude is conserved (central force)",
        value=(np.mean(Lmag) if np.all(np.isfinite(Lmag)) else None),
        residual=_fractional_drift(Lmag), tolerance=tol,
        provenance={"n_steps": int(len(traj.t)),
                    "quantity": "absL_peak_to_peak_over_mean"})
