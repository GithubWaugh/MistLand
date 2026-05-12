"""
snow.py
Phase xxx : Snow accumulation and melting.

Snow accumulation: when atmospheric temperature is below a threshold and there's enough mist, some mist turns into snow on the ground.
Conservation: mist lost == snow gained exactly.
Snow melting: when ground temperature is above 0°C, some snow turns into water.
Conservation: snow lost == water gained exactly.
"""


from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

def step(world: "World") -> None:
    cfg = world.config["snow"]

    snow_temp_threshold = cfg["snow_temp_threshold"]
    snow_melting_threshold = cfg["snow_melting_threshold"]
    mist_to_snow_rate = cfg["mist_to_snow_rate"]
    snow_melting_rate = cfg["snow_melting_rate"]

    f = world.front
    b = world.back

    # --- Snow accumulation ---
    can_snow = (b.atmo_temp < snow_temp_threshold) & (b.mist >= 0.01)
    snow_accumulation = np.where(can_snow, np.minimum(b.mist, mist_to_snow_rate), 0.0)
    snow_accumulation *= (1-b.ground_snow)  # Limit snow accumulation if there's already a lot of snow (max 1.0)
    b.mist -= snow_accumulation
    b.ground_snow += snow_accumulation

    # --- Snow melting ---
    can_melt = (b.ground_temp > snow_melting_threshold) & (b.ground_snow >0.0)
    snow_meltable = np.where(can_melt, np.minimum(b.ground_snow, snow_melting_rate), 0.0)
    melting_multiplier = np.minimum(10, np.maximum(0, b.atmo_temp) / 10)  # Melt faster if atmosphere is warmer (up to 10x at 10°C or above)
    melting_multiplier *= world.sun_exposition()  # Melt faster if sun exposition is higher (up to 1x at full sun, 0 at no sun)
    b.ground_snow -= snow_meltable * snow_melting_rate * melting_multiplier
    b.ground_water += snow_meltable * snow_melting_rate * melting_multiplier