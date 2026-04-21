"""
evaporation.py
Phase 5 : Evaporation — ground water → atmosphere mist.

Each tick, for cells where ground temperature exceeds evap_temp_threshold
and ground water is present, a fraction (evap_rate) of ground water
is added to a float accumulator (mist_accumulator).

When the accumulator reaches MIST_UNIT, one integer mist unit is created
and MIST_UNIT is subtracted from both the accumulator and ground_water.
This ensures exact water conservation : water only leaves the ground
when a full mist unit is formed.

Conservation : total_water = ground_water.sum() + mist.sum() * MIST_UNIT
               + mist_accumulator.sum()   must remain constant.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World


# One mist unit represents this much ground_water
MIST_UNIT = 0.1


def step(world: "World") -> None:
    cfg = world.config["water"]

    temp_threshold = cfg["evap_temp_threshold"]
    evap_rate      = cfg["evap_rate"]

    f = world.front
    b = world.back

    # --- Cells where evaporation can occur ---
    can_evaporate = (f.ground_temp > temp_threshold) & (f.ground_water > 0.0)

    # --- Water entering the accumulator this tick ---
    evaporated = np.where(
        can_evaporate,
        evap_rate * f.ground_water,
        0.0
    ).astype(np.float32)

    # Clamp : cannot accumulate more than available ground water
    evaporated = np.minimum(evaporated, f.ground_water)

    # --- Update accumulator ---
    new_accumulator = (f.mist_accumulator + evaporated).astype(np.float32)

    # --- Convert full units from accumulator to mist ---
    mist_gain = (new_accumulator / MIST_UNIT).astype(np.uint8)   # full units formed

    # Clamp mist gain so total mist does not exceed 7
    mist_gain = np.minimum(
        mist_gain,
        (7 - f.mist.astype(np.int16)).clip(0).astype(np.uint8)
    )

    # Water actually consumed = full units * MIST_UNIT
    water_consumed = (mist_gain * MIST_UNIT).astype(np.float32)

    # Remainder stays in accumulator (no water lost)
    b.mist_accumulator  = (new_accumulator - water_consumed).astype(np.float32)

    # --- Apply to back buffer ---
    # Water leaves ground only when it actually enters the accumulator
    b.ground_water = (f.ground_water - evaporated).astype(np.float32)
    b.mist = (f.mist.astype(np.int16) + mist_gain.astype(np.int16)).astype(np.uint8)