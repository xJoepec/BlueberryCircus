"""Binding potentials and their force Jacobians.

Each potential exposes ``force(x)`` and ``force_jacobian(x) = d force / d x``.
The Jacobian is required by the Landau--Lifshitz radiation-reaction reduction
(:mod:`dynamics`), which needs ``d F_ext / dt = J . v``.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import Units, SI


@dataclass
class Harmonic:
    """Isotropic harmonic binding ``U = 1/2 m omega0^2 r^2``."""
    omega0: float
    mass: float = 1.0

    def potential(self, x):
        x = np.asarray(x, float)
        return 0.5 * self.mass * self.omega0**2 * np.dot(x, x)

    def force(self, x):
        x = np.asarray(x, float)
        return -self.mass * self.omega0**2 * x

    def force_jacobian(self, x):
        return -self.mass * self.omega0**2 * np.eye(len(np.atleast_1d(x)))


@dataclass
class Coulomb:
    """Attractive Coulomb binding ``U = -Z k_e q^2 / r`` (electron-nucleus).

    A Plummer softening length ``softening`` (default 0) regularizes the r->0
    singularity for finite-step integration; set it well below the orbit scale.
    """
    Z: float = 1.0
    units: Units = SI
    softening: float = 0.0
    charge: float = None      # defaults to units.charge
    mass: float = None        # defaults to units.mass

    def __post_init__(self):
        if self.charge is None:
            self.charge = self.units.charge
        if self.mass is None:
            self.mass = self.units.mass
        self._coef = self.Z * self.units.k_e * self.charge**2

    def _rs(self, x):
        x = np.asarray(x, float)
        return np.sqrt(np.dot(x, x) + self.softening**2)

    def potential(self, x):
        return -self._coef / self._rs(x)

    def force(self, x):
        x = np.asarray(x, float)
        rs = self._rs(x)
        return -self._coef * x / rs**3

    def force_jacobian(self, x):
        x = np.asarray(x, float)
        rs = self._rs(x)
        I = np.eye(len(x))
        return -self._coef * (I / rs**3 - 3.0 * np.outer(x, x) / rs**5)


# ---------------------------------------------------------------------------
# Central-force family
# ---------------------------------------------------------------------------
class CentralPotential:
    """Base for isotropic central potentials ``U = U(r)``.

    A subclass supplies only the three *radial* functions ``_U(s)``,
    ``_dU(s) = dU/ds`` and ``_d2U(s) = d^2U/ds^2`` of the (optionally softened)
    radius ``s = sqrt(r^2 + softening^2)``. The Cartesian force and its Jacobian
    then follow from the chain rule, so every central potential inherits an
    analytic ``force_jacobian`` -- the quantity the Landau--Lifshitz
    radiation-reaction reduction needs (``dF/dt = J . v``).

    With ``F(x) = g(s) x`` and ``g(s) = -U'(s)/s`` the Jacobian is::

        J = g(s) I + (g'(s)/s) x x^T ,   g'(s) = (U'(s) - s U''(s)) / s^2 .

    The ``softening`` length regularizes the ``r -> 0`` core for singular laws
    (Coulomb-like power laws, Yukawa, Lennard-Jones) exactly as in
    :class:`Coulomb`; leave it at 0 for regular wells (harmonic, Morse, Duffing).
    """

    softening: float = 0.0

    def _s(self, x):
        x = np.asarray(x, float)
        return np.sqrt(np.dot(x, x) + self.softening**2)

    # radial functions -- implemented by subclasses
    def _U(self, s):   # pragma: no cover - abstract
        raise NotImplementedError
    def _dU(self, s):  # pragma: no cover - abstract
        raise NotImplementedError
    def _d2U(self, s):  # pragma: no cover - abstract
        raise NotImplementedError

    # Cartesian interface (shared by the integrator)
    def potential(self, x):
        return float(self._U(self._s(x)))

    def force(self, x):
        x = np.asarray(x, float)
        s = self._s(x)
        return -(self._dU(s) / s) * x

    def force_jacobian(self, x):
        x = np.asarray(x, float)
        s = self._s(x)
        g = -self._dU(s) / s
        gprime = (self._dU(s) - s * self._d2U(s)) / s**2
        return g * np.eye(len(x)) + (gprime / s) * np.outer(x, x)


@dataclass
class PowerLaw(CentralPotential):
    """Central power law ``U = coeff * r**p``.

    Generalizes the two hardcoded laws: ``p=2, coeff=1/2 m omega0^2`` reproduces
    :class:`Harmonic`; ``p=-1, coeff=-Z k_e q^2`` reproduces :class:`Coulomb`.
    Continuous ``p`` gives the ``F ~ r^(p-1)`` force families used as the
    inverse-dynamics generalization axis.
    """
    coeff: float
    p: float
    mass: float = 1.0
    softening: float = 0.0

    def _U(self, s):
        return self.coeff * s**self.p

    def _dU(self, s):
        return self.coeff * self.p * s**(self.p - 1.0)

    def _d2U(self, s):
        return self.coeff * self.p * (self.p - 1.0) * s**(self.p - 2.0)


@dataclass
class Yukawa(CentralPotential):
    """Screened Coulomb (Yukawa) ``U = -(g / r) exp(-r / lam)``.

    Attractive for ``g > 0``; ``lam`` is the screening length (``lam -> inf``
    recovers Coulomb). Set ``softening`` below the orbit scale to regularize the
    ``r -> 0`` core.
    """
    g: float = 1.0
    lam: float = 1.0
    mass: float = 1.0
    softening: float = 0.0

    def _U(self, s):
        return -self.g * np.exp(-s / self.lam) / s

    def _dU(self, s):
        e = np.exp(-s / self.lam)
        return self.g * e * (1.0 / (self.lam * s) + 1.0 / s**2)

    def _d2U(self, s):
        e = np.exp(-s / self.lam)
        L = self.lam
        return self.g * e * (-1.0 / (L**2 * s) - 2.0 / (L * s**2) - 2.0 / s**3)


@dataclass
class Morse(CentralPotential):
    """Radial Morse well ``U = De (1 - e^{-a(r - re)})^2 - De``.

    Minimum ``U(re) = -De``; ``U(inf) = 0``. A bounded, anharmonic binding well
    with a soft outer wall -- a qualitatively different force family from the
    power laws (finite dissociation energy).
    """
    De: float = 1.0
    a: float = 1.0
    re: float = 1.0
    mass: float = 1.0
    softening: float = 0.0

    def _w(self, s):
        return np.exp(-self.a * (s - self.re))

    def _U(self, s):
        w = self._w(s)
        return self.De * (1.0 - w)**2 - self.De

    def _dU(self, s):
        w = self._w(s)
        return 2.0 * self.De * self.a * w * (1.0 - w)

    def _d2U(self, s):
        w = self._w(s)
        return -2.0 * self.De * self.a**2 * w * (1.0 - 2.0 * w)


@dataclass
class AnharmonicOscillator(CentralPotential):
    """Isotropic Duffing well ``U = 1/2 m omega0^2 r^2 + 1/4 beta r^4``.

    ``beta > 0`` hardening, ``beta < 0`` softening; ``beta = 0`` recovers
    :class:`Harmonic`. The canonical nonlinear-oscillator testbed.
    """
    omega0: float = 1.0
    beta: float = 0.0
    mass: float = 1.0
    softening: float = 0.0

    def _U(self, s):
        return 0.5 * self.mass * self.omega0**2 * s**2 + 0.25 * self.beta * s**4

    def _dU(self, s):
        return self.mass * self.omega0**2 * s + self.beta * s**3

    def _d2U(self, s):
        return self.mass * self.omega0**2 + 3.0 * self.beta * s**2


@dataclass
class LennardJones(CentralPotential):
    """12-6 Lennard-Jones ``U = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]``.

    Stiff repulsive core plus a shallow attractive well (minimum at
    ``r = 2^(1/6) sigma``). The steep ``r^-12`` wall makes this the stress test
    for the integrator; keep the orbit away from the core or add ``softening``.
    """
    eps: float = 1.0
    sigma: float = 1.0
    mass: float = 1.0
    softening: float = 0.0

    def _U(self, s):
        u = self.sigma / s
        return 4.0 * self.eps * (u**12 - u**6)

    def _dU(self, s):
        u = self.sigma / s
        return -(4.0 * self.eps / s) * (12.0 * u**12 - 6.0 * u**6)

    def _d2U(self, s):
        u = self.sigma / s
        return (4.0 * self.eps / s**2) * (156.0 * u**12 - 42.0 * u**6)
