"""
atmosphere.py
Phase 6 : Atmospheric movements.

Wind transports mist and atmospheric temperature.

Temperature : net flux (antisymmetric) — conserved by construction.

Mist transport uses two separate schemes for stability :
  1. Isotropic diffusion — symmetric net flux, unconditionally stable.
     Spreads mist slowly in all directions regardless of wind.
  2. Wind advection — outflow proportional to wind strength and direction.
     Rate kept deliberately low to avoid divergence.

Both rates are tunable in config under "atmosphere" :
  mist_diffusion_rate  (default 0.02)
  mist_advection_rate  (default 0.05)

Conservation :
  Excess mist above 7.0 is converted to ground_water (instant precipitation)
  rather than being discarded.
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

    k_wind          = cfg["k_wind"]
    transport_rate  = cfg["wind_transport_rate"]
    diffusion_rate  = cfg.get("mist_diffusion_rate", 0.02)
    advection_rate  = cfg.get("mist_advection_rate", 0.05)

    f = world.front
    b = world.back

    # --- Wind strength toward each neighbour ---
    wind = []
    for dr, dc in _NEIGHBOURS:
        w = k_wind * (f.pressure - np.roll(np.roll(f.pressure, dr, axis=0), dc, axis=1))
        wind.append(np.clip(w, 0.0, 1.0).astype(np.float32))

    total_wind = np.minimum(sum(wind), 1.0).astype(np.float32)

    # Fraction of wind going to each neighbour
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

    # 1. Isotropic diffusion (symmetric net flux — unconditionally stable)
    new_mist = f.mist.copy()
    n = len(_NEIGHBOURS)
    for dr, dc in _NEIGHBOURS:
        neighbour_mist = np.roll(np.roll(f.mist, dr, axis=0), dc, axis=1)
        diff_flux = (diffusion_rate / n
                     * (f.mist - neighbour_mist)).astype(np.float32)
        new_mist -= diff_flux
        new_mist += np.roll(np.roll(diff_flux, -dr, axis=0), -dc, axis=1)

    # 2. Wind advection (directional outflow — small rate for stability)
    mist_outflow = (advection_rate * total_wind * f.mist).astype(np.float32)
    mist_outflow  = np.minimum(mist_outflow, f.mist)
    new_mist     -= mist_outflow
    for i, (dr, dc) in enumerate(_NEIGHBOURS):
        new_mist += np.roll(np.roll(
            (mist_outflow * fractions[i]).astype(np.float32),
            -dr, axis=0), -dc, axis=1)

    # --- Excess mist → ground water (conservation) ---
    excess_mist      = np.maximum(new_mist - 7.0, 0.0).astype(np.float32)
    excess_water     = (excess_mist * MIST_UNIT).astype(np.float32)
    new_mist         = np.minimum(new_mist, 7.0).astype(np.float32)
    new_ground_water = (f.ground_water + excess_water).astype(np.float32)

    # --- Apply ---
    b.atmo_temp    = new_atmo_temp.astype(np.float32)
    b.mist         = np.maximum(new_mist, 0.0).astype(np.float32)
    b.ground_water = new_ground_water