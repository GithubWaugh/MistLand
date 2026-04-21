"""
rain.py
Phase 8 : Rain — atmosphere mist → ground water.
  If atmospheric temperature falls below threshold
  and mist exceeds humidity threshold, it rains.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.world import World


def step(world: "World") -> None:
    cfg     = world.config["rain"]
    f       = world.front
    b       = world.back

    temp_threshold      = cfg["rain_temp_threshold"]
    humidity_threshold  = cfg["rain_humidity_threshold"]
    rain_rate           = cfg["rain_rate"]

    # TODO : implement rain
    pass
