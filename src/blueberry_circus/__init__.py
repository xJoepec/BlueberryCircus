"""BlueberryCircus -- a classical stochastic-electrodynamics (SED) simulator.

Nanarch Technologies, Inc.

A library in the lineage of Strawberry Fields / Mr Mustard / PennyLane, but for
*classical* stochastic electrodynamics: the dynamics of a charged point particle
bound by a potential and immersed in a classical zero-point-fluctuation
(ZPF) electromagnetic background, whose electric energy is stored capacitively
through the vacuum permittivity ``u_E = (1/2) eps0 <E^2>`` (Puthoff 1987;
Boyer 1975; Cole & Zou 2003; Nieuwenhuizen & Liska 2015).
"""
from .constants import (Units, SI, BOHR, EPS0, HBAR, C, E_CHARGE, M_E, K_E,
                        ALPHA, A0, radiation_reaction_time,
                        setterfield_rescale)
from . import (spectrum, oracles, observables, potentials, symplectic,
               rectification, tournament, conservation)
from .spectrum import rho, spectral_density_Ex, mode_density, mode_energy
from .zpf import ZPFBackground
from .dynamics import Particle, Trajectory, integrate
from .certify import Certificate, RULES, PASS, FAIL, NULL, audit_overclaim, \
    save_bundle, load_bundle
from .program import (Program, Harmonic, Coulomb, PowerLaw, Yukawa, Morse,
                      Anharmonic, LennardJones, ZPF, RadiationReaction,
                      Operation)
from .conservation import (energy_conservation_certificate,
                           angular_momentum_conservation_certificate)
from .engine import Engine, Result
from .tournament import (OrbitState, EnergyLedger, TournamentConfig,
                         HypothesisResult)

__version__ = "0.3.0"

__all__ = [
    "Units", "SI", "BOHR", "EPS0", "HBAR", "C", "E_CHARGE", "M_E", "K_E",
    "ALPHA", "A0", "radiation_reaction_time", "setterfield_rescale",
    "spectrum", "oracles", "observables", "potentials", "symplectic",
    "rectification", "tournament", "conservation", "rho", "spectral_density_Ex",
    "mode_density", "mode_energy",
    "ZPFBackground", "Particle", "Trajectory", "integrate", "Certificate",
    "RULES", "PASS", "FAIL", "NULL", "audit_overclaim", "save_bundle",
    "load_bundle", "Program", "Harmonic", "Coulomb", "PowerLaw", "Yukawa",
    "Morse", "Anharmonic", "LennardJones", "ZPF", "RadiationReaction",
    "Operation", "energy_conservation_certificate",
    "angular_momentum_conservation_certificate",
    "Engine", "Result", "OrbitState", "EnergyLedger",
    "TournamentConfig", "HypothesisResult", "__version__",
]
