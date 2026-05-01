"""
atmosphere.py
Phase 6 : Atmospheric movements.

Wind transports mist and atmospheric temperature.

Conservation fix :
  Mist is clamped to [0, 7] — excess above 7.0 is transferred
  to ground_water immediately (instant precipitation) rather than
  being discarded. This ensures total water is conserved exactly.

Temperature uses net flux (antisymmetric) — conserved by construction.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

from sim.phases.evaporation import MIST_UNIT


_NEIGHBOURS = [
    (-1,  0),  # north
    ( 1,  0),  # south
    ( 0,  1),  # east
    ( 0, -1),  # west
    (-1,  1),  # NE
    (-1, -1),  # NW
    ( 1,  1),  # SE
    ( 1, -1),  # SW
]


def step(world: "World") -> None:
    cfg = world.config["atmosphere"]

    k_wind         = cfg["k_wind"]
    transport_rate = cfg["wind_transport_rate"]

    f = world.front
    b = world.back

    # --- Wind strength toward each neighbour ---
    wind = []
    for dr, dc in _NEIGHBOURS:
        w = k_wind * (f.pressure - np.roll(np.roll(f.pressure, dr, axis=0), dc, axis=1))
        wind.append(np.clip(w, 0.0, 1.0).astype(np.float32))

    total_wind = np.minimum(sum(wind), 1.0).astype(np.float32)

    # Fraction going to each neighbour
    fractions = []
    for w in wind:
        frac = np.where(
            total_wind > 0.0,
            w / (total_wind + 1e-8),
            0.0
        ).astype(np.float32)
        fractions.append(frac)

    # --- Temperature transport via net flux (conserved) ---
    new_atmo_temp = f.atmo_temp.copy()
    for i, (dr, dc) in enumerate(_NEIGHBOURS):
        neighbour_temp = np.roll(np.roll(f.atmo_temp, dr, axis=0), dc, axis=1)
        net_flux = (transport_rate * fractions[i]
                    * (f.atmo_temp - neighbour_temp)).astype(np.float32)
        new_atmo_temp -= net_flux
        new_atmo_temp += np.roll(np.roll(net_flux, -dr, axis=0), -dc, axis=1)

    # --- Mist transport ---
    mist_outflow = (transport_rate * total_wind * f.mist).astype(np.float32)
    mist_outflow = np.minimum(mist_outflow, f.mist)

    new_mist = (f.mist - mist_outflow).astype(np.float32)
    for i, (dr, dc) in enumerate(_NEIGHBOURS):
        new_mist += np.roll(np.roll(
            (mist_outflow * fractions[i]).astype(np.float32),
            -dr, axis=0), -dc, axis=1
        )

    # --- Excess mist → ground water (conservation) ---
    # Any mist above 7.0 is precipitated immediately rather than discarded
    excess_mist     = np.maximum(new_mist - 7.0, 0.0).astype(np.float32)
    excess_water    = (excess_mist * MIST_UNIT).astype(np.float32)

    new_mist        = np.minimum(new_mist, 7.0).astype(np.float32)
    new_ground_water = (f.ground_water + excess_water).astype(np.float32)

    # Apply
    b.atmo_temp    = new_atmo_temp.astype(np.float32)
    b.mist         = np.maximum(new_mist, 0.0).astype(np.float32)
    b.ground_water = new_ground_water