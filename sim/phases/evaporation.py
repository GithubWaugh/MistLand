"""
evaporation.py
Phase 5 : Evaporation — ground water → atmosphere mist.
  If ground temperature exceeds threshold and ground water > 0,
  a fraction transfers to mist.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.world import World


def step(world: "World") -> None:
    cfg     = world.config["water"]
    f       = world.front
    b       = world.back

    temp_threshold  = cfg["evap_temp_threshold"]
    evap_rate       = cfg["evap_rate"]

    # TODO : implement evaporation
    pass
