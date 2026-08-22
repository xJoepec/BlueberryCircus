"""Central-force potential family: declarative programs + conservation certificates.

Runs a near-circular orbit under each new potential with radiation reaction off
(a closed conservative system) and prints the re-checkable energy and
angular-momentum conservation certificates. Because the ZPF background is a pure
function of its seed, swapping the potential while holding the seed fixed gives a
*matched-seed counterfactual*: identical driving noise, one edited force law.

    python examples/central_forces.py
"""
import numpy as np

import blueberry_circus as bc
from blueberry_circus.dynamics import Particle, integrate
from blueberry_circus.conservation import (
    energy_conservation_certificate, angular_momentum_conservation_certificate,
)

U = bc.Units.scaled(gamma_over_omega0=0.05, omega0=1.0)
P = Particle(U.charge, U.mass)

POTENTIALS = {
    "PowerLaw(p=-1, Kepler)": bc.potentials.PowerLaw(coeff=-0.7, p=-1.0, softening=1e-3),
    "PowerLaw(p=2.5)":        bc.potentials.PowerLaw(coeff=0.4, p=2.5),
    "Yukawa":                 bc.potentials.Yukawa(g=0.8, lam=3.0, softening=1e-3),
    "Morse":                  bc.potentials.Morse(De=3.0, a=0.8, re=1.0),
    "Anharmonic(Duffing)":    bc.potentials.AnharmonicOscillator(omega0=1.0, beta=0.3),
    "LennardJones":           bc.potentials.LennardJones(eps=1.0, sigma=0.6),
}


def orbit(pot, r0=1.0, tmax=120.0, dt=0.005):
    Fmag = np.linalg.norm(pot.force([r0, 0, 0]))
    v = np.sqrt(Fmag * r0 / P.mass)
    t = np.arange(0.0, tmax, dt)
    return integrate(field=None, potential=pot, particle=P, t_grid=t,
                     x0=[r0, 0, 0], v0=[0, v, 0], rr="none", units=U, dipole=False)


def main():
    print(f"{'potential':24s} {'E drift':>12s} {'|L| drift':>12s}  verdicts")
    for name, pot in POTENTIALS.items():
        tr = orbit(pot)
        e = energy_conservation_certificate(tr, pot, P, tol=1e-6)
        l = angular_momentum_conservation_certificate(tr, P, tol=1e-8)
        print(f"{name:24s} {e.residual:12.2e} {l.residual:12.2e}  "
              f"[E:{e.recheck()}] [L:{l.recheck()}]")

    # matched-seed counterfactual teaser: same ZPF seed, two force laws.
    print("\nmatched-seed counterfactual (same noise, edited law):")
    for law in (bc.Harmonic(omega0=1.0), bc.Anharmonic(omega0=1.0, beta=0.4)):
        prog = bc.Program(n_particles=1, units=U)
        with prog.context as q:
            law                                                | q[0]
            bc.ZPF(band=(0.3, 3.0), n_modes=64, seed=7,
                   mode="one_dimensional", axis=0)             | q[0]
            bc.RadiationReaction("landau_lifshitz")            | q[0]
        res = bc.Engine(dt=0.02, t_max=60.0).run(prog, x0=[0.1, 0, 0], v0=[0, 0, 0])
        print(f"  {type(law).__name__:12s} <x^2>={res.observables['position_variance'][0]:.4e}")


if __name__ == "__main__":
    main()
