# Central-force potentials

BlueberryCircus began with two hardcoded binding laws — `Harmonic` and
`Coulomb`. This module generalizes the potential layer into an extensible
**central-force family** while keeping the engine, the radiation-reaction
reduction, and the certificate discipline untouched.

## The abstraction

A potential is any object exposing `potential(x)`, `force(x)`, and
`force_jacobian(x)`. The Jacobian is not optional: the Landau–Lifshitz
radiation-reaction reduction needs `dF/dt = J · v`. To make new laws cheap and
correct, `CentralPotential` derives all three Cartesian quantities from three
*radial* functions of the (optionally softened) radius `s = √(r² + softening²)`:

| supply | meaning |
|---|---|
| `_U(s)`   | potential energy `U(r)` |
| `_dU(s)`  | `U'(r)` |
| `_d2U(s)` | `U''(r)` |

With `F(x) = g(s) x`, `g(s) = −U'(s)/s`, the analytic Jacobian is

```
J = g(s) I + (g'(s)/s) x xᵀ ,   g'(s) = (U'(s) − s U''(s)) / s² .
```

Every subclass is a handful of lines and inherits an exact `force_jacobian`.

## The families

| class | `U(r)` | notes |
|---|---|---|
| `PowerLaw(coeff, p)` | `coeff · rᵖ` | continuous exponent; `p=2` → harmonic, `p=−1` → Coulomb |
| `Yukawa(g, lam)` | `−(g/r) e^{−r/λ}` | screened Coulomb (`λ→∞` → Coulomb) |
| `Morse(De, a, re)` | `De(1−e^{−a(r−re)})² − De` | bounded well, finite dissociation energy |
| `AnharmonicOscillator(omega0, beta)` | `½ m ω₀² r² + ¼ β r⁴` | isotropic Duffing (`β=0` → harmonic) |
| `LennardJones(eps, sigma)` | `4ε[(σ/r)¹² − (σ/r)⁶]` | stiff repulsive core; use `softening` or keep the orbit off the core |

Singular laws take a Plummer `softening` length that regularizes the `r→0` core,
identical in spirit to `Coulomb.softening`.

## Declarative use

The new operations plug into the existing `op | q[i]` program model. Potential
collection in `Program.compile()` is now keyed on the `_PotentialOp` marker, so
future potentials are picked up automatically (no more hardcoded tuple).

```python
import blueberry_circus as bc

prog = bc.Program(n_particles=1)
with prog.context as q:
    bc.Yukawa(g=0.8, lam=2.0, softening=1e-3)          | q[0]
    bc.ZPF(band=(0.3, 3.0), n_modes=200, seed=0)       | q[0]
    bc.RadiationReaction("landau_lifshitz")            | q[0]
result = bc.Engine(dt=0.02, t_max=600).run(prog, x0=[0, 0, 0], v0=[0, 0, 0])
```

## Conservation certificates

`blueberry_circus.conservation` adds two re-checkable certificates for closed
conservative systems (`rr="none"`, no field):

- `energy_conservation_certificate(traj, potential, particle, tol)`
- `angular_momentum_conservation_certificate(traj, particle, tol)`

Both encode the fractional peak-to-peak drift of the invariant against the
canonical `residual_le_tol` rule, so an under-resolved integration or a tampered
trajectory re-derives `FAIL`. For a central force `|L|` conservation is the
signature of centrality, and energy conservation cross-checks that a new
potential's analytic `force` is consistent with its `potential` energy — purely
from the integrated dynamics. See `examples/central_forces.py`; all six families
conserve both invariants to ≲10⁻¹¹ at `dt = 5·10⁻³`.

## Correctness

`tests/test_central_potentials.py` pins the family three independent ways:

1. **Finite differences** — analytic `force` and `force_jacobian` match central
   differences of `potential` and `force`; the Jacobian is symmetric (curl-free).
2. **Reduction oracles** — `PowerLaw` reproduces `Harmonic` (`p=2`) and `Coulomb`
   (`p=−1`) to floating-point closeness.
3. **Conservation** — closed orbits pass the energy and angular-momentum
   certificates, and a real drift `FAIL`s an impossibly tight tolerance.
