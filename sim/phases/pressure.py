"""
pressure.py
Phase 2 : Atmospheric pressure computation.

Formula :
    pressure(cell) = P_base
                   - k_temp * (atmo_temp - temp_ref)
                   - k_alt  * altitude 

- Cell at temp_ref → pressure = P_base (neutral)
- Hot air  → low pressure  (air expands and rises)
- Cold air → high pressure (air contracts and sinks)
- High alt → low pressure  (less air mass above)

A damping coefficient smooths the transition between ticks
to prevent numerical oscillation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

def step(world: "World") -> None:
    cfg = world.config["atmosphere"]

    P_base   = cfg["P_base"]
    k_temp   = cfg["k_temp"]
    k_alt    = cfg["k_alt"]
    temp_ref = cfg["temp_ref"]
    damping  = cfg["pressure_damping"]

    surface_altitude = world.surface_altitude()  


    # Target pressure (vectorized over entire grid)
    target = (
        P_base
        - k_temp * (world.front.atmo_temp - temp_ref)
        - k_alt  * surface_altitude
        + 0*world.front.mist * (1+cfg.get("mist_pressure_factor", 0.0))/7  # Optional mist contribution to pressure
    ).astype(np.float32)

    # Smooth transition toward target
    # damping = 1.0 → instant jump
    # damping = 0.1 → slow drift (more stable)
    world.back.pressure = (
        world.front.pressure + damping * (target - world.front.pressure)
    ).astype(np.float32)