"""
vegetation.py
Phase 3 : Vegetation growth and devolution.
  - Evolves upward if water, temperature and nutriments conditions are met.
  - Devolves downward if conditions are not met.
  - Growth occurs at most once every vegetation_growth_period ticks.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.world import World


def step(world: "World") -> None:
    cfg     = world.config["vegetation"]
    f       = world.front
    b       = world.back

    growth_period       = cfg["growth_period_ticks"]
    water_min           = cfg["water_min"]
    temp_min            = cfg["temp_min"]
    temp_max            = cfg["temp_max"]
    nutriments_min      = cfg["nutriments_min"]

    # TODO : implement vegetation growth / devolution
    pass
