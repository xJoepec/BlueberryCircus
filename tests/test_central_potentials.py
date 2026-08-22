"""Tests for the central-force potential family.

Correctness is pinned three independent ways:

* analytic ``force`` and ``force_jacobian`` agree with central finite differences
  of ``potential`` and ``force`` (so the RR Jacobian is trustworthy);
* the general :class:`PowerLaw` reproduces the hardcoded :class:`Harmonic` and
  :class:`Coulomb` bit-closely (a reduction oracle);
* closed conservative orbits self-certify energy and angular-momentum
  conservation via :mod:`blueberry_circus.conservation`.
"""
import numpy as np
import pytest

import blueberry_circus as bc
from blueberry_circus.constants import Units
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.potentials import (
    Harmonic, Coulomb, PowerLaw, Yukawa, Morse, AnharmonicOscillator,
    LennardJones,
)
from blueberry_circus.conservation import (
    energy_conservation_certificate, angular_momentum_conservation_certificate,
)
from blueberry_circus.certify import PASS

U = Units.scaled(gamma_over_omega0=0.05, omega0=1.0)
P = Particle(U.charge, U.mass)

# Evaluation points kept away from the r->0 core (softening handles the rest).
PTS = [np.array([0.9, 0.0, 0.0]),
       np.array([0.7, -0.4, 0.5]),
       np.array([-0.6, 0.8, -0.3]),
       np.array([1.3, 0.2, -0.9])]

# One representative instance of each family (softening where the core is steep).
INSTANCES = [
    PowerLaw(coeff=0.5 * 1.3 * 0.9**2, p=2.0, mass=1.3),   # harmonic-like
    PowerLaw(coeff=-0.7, p=-1.0, softening=1e-3),          # Coulomb-like
    PowerLaw(coeff=0.4, p=2.5, mass=1.0),                  # fractional exponent
    Yukawa(g=0.8, lam=1.5, softening=1e-3),
    Morse(De=1.2, a=1.1, re=1.0),
    AnharmonicOscillator(omega0=0.9, beta=0.5, mass=1.1),
    LennardJones(eps=1.0, sigma=0.6),                      # well near r=0.67
]


def _num_grad_U(pot, x, h=1e-6):
    g = np.zeros(3)
    for i in range(3):
        d = np.zeros(3); d[i] = h
        g[i] = (pot.potential(x + d) - pot.potential(x - d)) / (2 * h)
    return g


def _num_jac_F(pot, x, h=1e-6):
    J = np.zeros((3, 3))
    for j in range(3):
        d = np.zeros(3); d[j] = h
        J[:, j] = (pot.force(x + d) - pot.force(x - d)) / (2 * h)
    return J


@pytest.mark.parametrize("pot", INSTANCES)
def test_force_is_minus_grad_potential(pot):
    for x in PTS:
        assert np.allclose(pot.force(x), -_num_grad_U(pot, x),
                           rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("pot", INSTANCES)
def test_force_jacobian_matches_finite_difference(pot):
    for x in PTS:
        assert np.allclose(pot.force_jacobian(x), _num_jac_F(pot, x),
                           rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("pot", INSTANCES)
def test_jacobian_is_symmetric(pot):
    # A conservative (curl-free) force has a symmetric Jacobian (Hessian of -U).
    for x in PTS:
        J = pot.force_jacobian(x)
        assert np.allclose(J, J.T, atol=1e-10)


def test_powerlaw_reduces_to_harmonic():
    m, w = 1.3, 0.9
    pl = PowerLaw(coeff=0.5 * m * w**2, p=2.0, mass=m)
    ha = Harmonic(w, mass=m)
    for x in PTS:
        assert np.isclose(pl.potential(x), ha.potential(x))
        assert np.allclose(pl.force(x), ha.force(x))
        assert np.allclose(pl.force_jacobian(x), ha.force_jacobian(x))


def test_powerlaw_reduces_to_coulomb():
    coul = Coulomb(Z=1.0, units=U, charge=U.charge, mass=U.mass)
    pl = PowerLaw(coeff=-coul._coef, p=-1.0)
    for x in PTS:
        assert np.isclose(pl.potential(x), coul.potential(x))
        assert np.allclose(pl.force(x), coul.force(x))
        assert np.allclose(pl.force_jacobian(x), coul.force_jacobian(x))


def _circular_orbit(pot, r0=1.0, tmax=120.0, dt=0.005):
    """Integrate a near-circular orbit in the plane for a central potential."""
    Fmag = np.linalg.norm(pot.force([r0, 0, 0]))
    v = np.sqrt(Fmag * r0 / P.mass)                # circular-orbit speed
    t = np.arange(0.0, tmax, dt)
    return integrate(field=None, potential=pot, particle=P, t_grid=t,
                     x0=[r0, 0, 0], v0=[0, v, 0], rr="none", units=U,
                     dipole=False)


@pytest.mark.parametrize("pot", [
    PowerLaw(coeff=-0.7, p=-1.0, softening=1e-3),
    Yukawa(g=0.8, lam=3.0, softening=1e-3),
    AnharmonicOscillator(omega0=1.0, beta=0.3),
    Morse(De=3.0, a=0.8, re=1.0),
])
def test_closed_orbit_conserves_energy_and_angular_momentum(pot):
    tr = _circular_orbit(pot)
    assert np.all(np.isfinite(tr.x))
    e_cert = energy_conservation_certificate(tr, pot, P, tol=1e-6)
    l_cert = angular_momentum_conservation_certificate(tr, P, tol=1e-8)
    assert e_cert.recheck() == PASS, f"energy drift residual={e_cert.residual:.2e}"
    assert l_cert.recheck() == PASS, f"|L| drift residual={l_cert.residual:.2e}"


def test_conservation_certificate_fails_on_impossible_tolerance():
    # A real (small but nonzero) drift must FAIL an absurdly tight tolerance:
    # guards against a certificate that trivially passes everything.
    tr = _circular_orbit(AnharmonicOscillator(omega0=1.0, beta=0.3))
    cert = energy_conservation_certificate(tr, AnharmonicOscillator(
        omega0=1.0, beta=0.3), P, tol=1e-18)
    assert cert.recheck() != PASS


@pytest.mark.parametrize("pot", [
    Morse(De=2.0, a=1.0, re=1.0),
    AnharmonicOscillator(omega0=1.0, beta=0.5),
    Yukawa(g=1.0, lam=2.0, softening=1e-2),
])
def test_radiation_reaction_stays_finite(pot):
    # The analytic Jacobian feeds the Landau-Lifshitz term; integration must not
    # blow up for the new families.
    Ud = Units.scaled(gamma_over_omega0=0.02, omega0=1.0)
    Pd = Particle(Ud.charge, Ud.mass)
    t = np.arange(0.0, 60.0, 0.01)
    tr = integrate(field=None, potential=pot, particle=Pd, t_grid=t,
                   x0=[1.0, 0, 0], v0=[0, 0.5, 0], rr="landau_lifshitz",
                   units=Ud, dipole=False)
    assert np.all(np.isfinite(tr.x)) and np.all(np.isfinite(tr.v))


def test_declarative_program_runs_new_potential():
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Yukawa(g=0.8, lam=2.0, softening=1e-3)              | q[0]
        bc.ZPF(band=(0.3, 3.0), n_modes=64, seed=1,
               mode="one_dimensional", axis=0)                 | q[0]
        bc.RadiationReaction("landau_lifshitz")                | q[0]
    res = bc.Engine(backend="numpy", dt=0.02, t_max=40.0).run(
        prog, x0=[1.0, 0, 0], v0=[0, 0.3, 0])
    assert res.observables["trajectory_finite"] is True
    assert res.certificates[0].recheck() == PASS


def test_program_still_requires_exactly_one_potential():
    prog = bc.Program(n_particles=1, units=U)
    with prog.context as q:
        bc.Harmonic(omega0=1.0)                                | q[0]
        bc.Morse(De=1.0, a=1.0, re=1.0)                        | q[0]
    with pytest.raises(ValueError):
        prog.compile(0)
