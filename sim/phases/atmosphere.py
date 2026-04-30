"""
atmosphere.py
Phase 6 : Atmospheric movements.

Wind is derived from the pressure gradient between neighbouring cells :
    wind(A → B) = k_wind × (pressure(A) - pressure(B))

Each tick, wind transports the cell's atmospheric content
(mist and temperature) to neighbouring cells, proportionally
to wind strength toward each neighbour.

Conservation :
  - Total mist (integer units) is conserved exactly.
  - Total atmo_temp is conserved (what leaves one cell enters another).

Note : mist is an integer [0..7]. Transport uses a float accumulator
(atmo_mist_out) to decide how many full units actually move,
avoiding fractional mist loss.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

from sim.phases.evaporation import MIST_UNIT


_NEIGHBOURS = [
    (0,  1),   # north
    (0, -1),   # south
    (1, -1),   # east
    (1,  1),   # west
]


def step(world: "World") -> None:
    cfg = world.config["atmosphere"]

    k_wind         = cfg["k_wind"]
    transport_rate = cfg["wind_transport_rate"]

    f = world.front
    b = world.back

    # --- Wind strength toward each neighbour ---
    # wind(A→B) = k_wind * (pressure(A) - pressure(B))
    # Clamped to [0, 1] : only outflow (negative gradient = no wind that way)
    wind = []
    for axis, shift in _NEIGHBOURS:
        w = k_wind * (f.pressure - np.roll(f.pressure, shift, axis=axis))
        wind.append(np.clip(w, 0.0, 1.0).astype(np.float32))

    # Total outgoing wind (sum over all neighbours)
    total_wind_raw = np.add.reduce(wind).astype(np.float32)

    # Clamp total to 1.0 : a cell cannot send more than 100% of its content
    total_wind = np.minimum(total_wind_raw, 1.0)

    # --- Fraction of content going to each neighbour ---
    # Normalize by the pre-clamp sum so fractions always sum to 1.0,
    # even when total_wind_raw > 1.0.  Using total_wind here would make
    # fractions sum to total_wind_raw, creating energy/mist out of nothing.
    fractions = []
    for w in wind:
        frac = np.where(
            total_wind_raw > 0.0,
            w / (total_wind_raw + 1e-8),
            0.0
        ).astype(np.float32)
        fractions.append(frac)

    # --- Atmosphere temperature transport ---
    # Each cell sends transport_rate * total_wind * atmo_temp to neighbours
    temp_outflow = (transport_rate * total_wind * f.atmo_temp).astype(np.float32)

    new_atmo_temp = (f.atmo_temp - temp_outflow).astype(np.float32)

    for i, (axis, shift) in enumerate(_NEIGHBOURS):
        temp_to_neighbour = (temp_outflow * fractions[i]).astype(np.float32)
        new_atmo_temp += np.roll(temp_to_neighbour, -shift, axis=axis)

    # --- Mist transport ---
    # Use a float accumulator to avoid fractional mist loss.
    # mist_accumulator already exists in buffers for evaporation —
    # we reuse it here as a shared accumulator for all atmospheric mist movement.

    # Float mist available (integer mist + existing accumulator)
    float_mist = (f.mist.astype(np.float32) * MIST_UNIT
                  + f.mist_accumulator)

    # Total float mist leaving this cell
    mist_outflow = (transport_rate * total_wind * float_mist).astype(np.float32)
    mist_outflow = np.minimum(mist_outflow, float_mist)   # cannot send more than available

    new_float_mist = (float_mist - mist_outflow).astype(np.float32)

    for i, (axis, shift) in enumerate(_NEIGHBOURS):
        mist_to_neighbour = (mist_outflow * fractions[i]).astype(np.float32)
        new_float_mist += np.roll(mist_to_neighbour, -shift, axis=axis)

    # Convert back to integer mist + accumulator
    new_mist        = (new_float_mist / MIST_UNIT).astype(np.uint8)
    new_mist        = np.clip(new_mist, 0, 7).astype(np.uint8)
    new_accumulator = (new_float_mist - new_mist.astype(np.float32) * MIST_UNIT).astype(np.float32)
    new_accumulator = np.maximum(new_accumulator, 0.0)

    # --- Apply to back buffer ---
    b.atmo_temp        = new_atmo_temp
    b.mist             = new_mist
    b.mist_accumulator = new_accumulator