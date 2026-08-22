"""Declarative program model (Strawberry-Fields / PennyLane style).

A :class:`Program` is a register of charged particles plus a list of operations
applied with the ``op | q[i]`` syntax::

    import blueberry_circus as bc
    prog = bc.Program(n_particles=1)
    with prog.context as q:
        bc.Coulomb(Z=1)                         | q[0]
        bc.ZPF(band=(w_lo, w_hi), n_modes=400)  | q[0]
        bc.RadiationReaction("landau_lifshitz") | q[0]
    result = bc.Engine(dt=dt, t_max=T).run(prog, x0=x0, v0=v0)

Operations are thin declarations; the :class:`~blueberry_circus.engine.Engine`
compiles them into the concrete potential / field / radiation-reaction model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import contextlib

from .constants import Units, SI
from . import potentials as _pot
from .zpf import ZPFBackground


class Operation:
    """Base op. ``op | regref`` appends the op to the bound program."""
    def __or__(self, regref: "RegRef"):
        regref.prog._append(self, regref.index)
        return self


class _PotentialOp(Operation):
    """Marker for ops that compile to a single binding potential.

    ``compile()`` collects potential ops by this base rather than a hardcoded
    ``(Harmonic, Coulomb)`` tuple, so new potential families are picked up
    automatically. Every subclass must provide ``build_potential(units, mass)``.
    """


@dataclass
class Harmonic(_PotentialOp):
    omega0: float

    def build_potential(self, units: Units, mass: float):
        return _pot.Harmonic(self.omega0, mass=mass)


@dataclass
class Coulomb(_PotentialOp):
    Z: float = 1.0
    softening: float = 0.0

    def build_potential(self, units: Units, mass: float):
        return _pot.Coulomb(Z=self.Z, units=units, softening=self.softening,
                            mass=mass)


@dataclass
class PowerLaw(_PotentialOp):
    """Central power law ``U = coeff * r**p`` (see :class:`potentials.PowerLaw`)."""
    coeff: float
    p: float
    softening: float = 0.0

    def build_potential(self, units: Units, mass: float):
        return _pot.PowerLaw(coeff=self.coeff, p=self.p, mass=mass,
                             softening=self.softening)


@dataclass
class Yukawa(_PotentialOp):
    """Screened Coulomb ``U = -(g/r) exp(-r/lam)``."""
    g: float = 1.0
    lam: float = 1.0
    softening: float = 0.0

    def build_potential(self, units: Units, mass: float):
        return _pot.Yukawa(g=self.g, lam=self.lam, mass=mass,
                           softening=self.softening)


@dataclass
class Morse(_PotentialOp):
    """Radial Morse well ``U = De (1 - e^{-a(r-re)})^2 - De``."""
    De: float = 1.0
    a: float = 1.0
    re: float = 1.0
    softening: float = 0.0

    def build_potential(self, units: Units, mass: float):
        return _pot.Morse(De=self.De, a=self.a, re=self.re, mass=mass,
                          softening=self.softening)


@dataclass
class Anharmonic(_PotentialOp):
    """Isotropic Duffing well ``U = 1/2 m w0^2 r^2 + 1/4 beta r^4``."""
    omega0: float = 1.0
    beta: float = 0.0
    softening: float = 0.0

    def build_potential(self, units: Units, mass: float):
        return _pot.AnharmonicOscillator(omega0=self.omega0, beta=self.beta,
                                         mass=mass, softening=self.softening)


@dataclass
class LennardJones(_PotentialOp):
    """12-6 Lennard-Jones ``U = 4 eps [(sigma/r)^12 - (sigma/r)^6]``."""
    eps: float = 1.0
    sigma: float = 1.0
    softening: float = 0.0

    def build_potential(self, units: Units, mass: float):
        return _pot.LennardJones(eps=self.eps, sigma=self.sigma, mass=mass,
                                 softening=self.softening)


@dataclass
class ZPF(Operation):
    band: tuple
    n_modes: int = 200
    seed: int = 0
    mode: str = "isotropic_3d"     # or "one_dimensional"
    axis: int = 0
    log_spaced: bool = False

    def build_field(self, units: Units) -> ZPFBackground:
        lo, hi = self.band
        if self.mode == "isotropic_3d":
            return ZPFBackground.isotropic_3d(lo, hi, self.n_modes, self.seed,
                                              units, self.log_spaced)
        elif self.mode == "one_dimensional":
            return ZPFBackground.one_dimensional(lo, hi, self.n_modes, self.seed,
                                                 units, self.axis, self.log_spaced)
        raise ValueError(f"unknown ZPF mode {self.mode!r}")


@dataclass
class RadiationReaction(Operation):
    model: str = "landau_lifshitz"

    def build(self) -> str:
        """Resolve the radiation-reaction model, rejecting the bare-AL runaway form.

        Only the non-runaway Landau--Lifshitz reduction (or ``none``) is allowed;
        the literal Abraham--Lorentz third-derivative form is never integrated.
        """
        if self.model not in ("landau_lifshitz", "none"):
            raise ValueError(
                f"unknown radiation-reaction model {self.model!r}; "
                "use 'landau_lifshitz' or 'none' (bare Abraham-Lorentz is a "
                "runaway trap and is not integrable here)")
        return self.model


@dataclass
class RegRef:
    prog: "Program"
    index: int


class _Register:
    def __init__(self, prog, n):
        self.prog = prog
        self.refs = [RegRef(prog, i) for i in range(n)]

    def __getitem__(self, i):
        return self.refs[i]

    def __iter__(self):
        return iter(self.refs)


@dataclass
class Program:
    n_particles: int = 1
    units: Units = SI
    ops: List = field(default_factory=list)   # list of (op, index)

    def __post_init__(self):
        self._register = _Register(self, self.n_particles)

    def _append(self, op, index):
        self.ops.append((op, index))

    @property
    @contextlib.contextmanager
    def context(self):
        yield self._register

    def ops_for(self, index: int):
        return [op for (op, idx) in self.ops if idx == index]

    def compile(self, index: int = 0) -> "CompiledProgram":
        """Lower the declared ops into a :class:`CompiledProgram` via named,
        fail-closed passes (the PennyLane transform-as-pass model). Each pass is
        recorded for inspectability; an ambiguous program raises *here* -- the
        interpreter can no longer silently keep only the last op of a kind.
        """
        passes: List = []
        if not (0 <= index < self.n_particles):
            raise ValueError(f"particle index {index} out of range "
                             f"[0, {self.n_particles})")
        passes.append("validate_single_particle")
        ops = self.ops_for(index)

        pots = [op for op in ops if isinstance(op, _PotentialOp)]
        if len(pots) != 1:
            raise ValueError(f"collect_potential: exactly one potential required "
                             f"for particle {index}, got {len(pots)}")
        potential = pots[0].build_potential(self.units, self.units.mass)
        passes.append("collect_potential")

        zpfs = [op for op in ops if isinstance(op, ZPF)]
        if len(zpfs) > 1:
            raise ValueError(f"collect_field: at most one ZPF field for particle "
                             f"{index}, got {len(zpfs)}")
        field_obj = zpfs[0].build_field(self.units) if zpfs else None
        passes.append("collect_field")

        reactions = [op for op in ops if isinstance(op, RadiationReaction)]
        if len(reactions) > 1:
            raise ValueError(f"resolve_reaction: at most one RadiationReaction for "
                             f"particle {index}, got {len(reactions)}")
        rr = reactions[0].build() if reactions else "none"
        passes.append("resolve_reaction")

        handled = (_PotentialOp, ZPF, RadiationReaction)
        unknown = [op for op in ops if not isinstance(op, handled)]
        if unknown:
            raise TypeError(f"compile: unhandled operation(s) {unknown!r}")
        passes.append("validate_ops_exhaustive")

        return CompiledProgram(units=self.units, potential=potential,
                               field=field_obj, rr=rr,
                               n_particles=self.n_particles, index=index,
                               passes=passes)


@dataclass
class CompiledProgram:
    """The lowered, backend-ready form of a :class:`Program` for one particle.

    ``passes`` records the named compile passes that produced it (inspectable).
    """
    units: Units
    potential: object
    field: object
    rr: str
    n_particles: int
    index: int
    passes: List = field(default_factory=list)
