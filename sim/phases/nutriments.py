"""
nutriments.py
Phase 4 : Nutriment diffusion to neighbours.
  Simulates microfauna movement (insects, worms, etc.)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.world import World


def step(world: "World") -> None:
    cfg     = world.config["nutriments"]
    f       = world.front
    b       = world.back

    diffusion_rate  = cfg["diffusion_rate"]

    # TODO : implement nutriment diffusion
    pass
