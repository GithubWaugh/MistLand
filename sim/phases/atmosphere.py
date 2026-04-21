"""
atmosphere.py
Phase 6 : Atmospheric movements.
  Wind (derived from pressure gradient) transports mist and temperature
  to neighbouring cells, proportionally to wind strength.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.world import World


def step(world: "World") -> None:
    cfg     = world.config["atmosphere"]
    f       = world.front
    b       = world.back

    k_wind          = cfg["k_wind"]
    transport_rate  = cfg["wind_transport_rate"]

    # TODO : implement atmospheric transport
    pass
