"""
pressure.py
Phase 2 : Atmospheric pressure computation.
  pressure(cell) = P_base - k_temp * atmo_temp + k_alt * altitude
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.world import World


def step(world: "World") -> None:
    cfg     = world.config["atmosphere"]
    f       = world.front
    b       = world.back

    P_base          = cfg["P_base"]
    k_temp          = cfg["k_temp"]
    k_alt           = cfg["k_alt"]
    pressure_damping = cfg["pressure_damping"]

    # TODO : implement pressure computation
    pass
