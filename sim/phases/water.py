"""
water.py
Phase 7 : Water movements — runoff and flooding.
  - High-ground cells distribute water to lower neighbours,
    proportionally to altitude delta.
  - Sand and soil retain a minimum amount of water.
  - Cells exceeding flooding threshold become lakes,
    generating nutriments for their neighbours.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.world import World


def step(world: "World") -> None:
    cfg         = world.config["water"]
    base_cfg    = world.config["base_types"]
    f           = world.front
    b           = world.back

    runoff_rate = cfg["runoff_rate"]

    # TODO : implement runoff and flooding
    pass
