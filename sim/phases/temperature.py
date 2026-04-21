"""
temperature.py
Phase 1 : Temperature propagation.
  - Ground temp radiates to neighbours and to atmosphere.
  - Atmosphere temp radiates to ground.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.world import World


def step(world: "World") -> None:
    cfg     = world.config["temperature"]
    f       = world.front   # read
    b       = world.back    # write

    gn_rate = cfg["ground_to_neighbour_rate"]
    ga_rate = cfg["ground_to_atmosphere_rate"]
    ag_rate = cfg["atmosphere_to_ground_rate"]

    # TODO : implement temperature propagation
    pass
