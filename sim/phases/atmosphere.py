"""
atmosphere.py
Phase 6 : Atmospheric movements.

Wind transports mist and atmospheric temperature.
Wind direction and strength now come from the persistent wind vector field
(wind_x, wind_y) computed by wind.py, rather than from the instantaneous
pressure gradient. This allows vortices, fronts, and memory effects.

Mist transport :
  1. Isotropic diffusion — symmetric, unconditionally stable.
  2. Wind advection      — directional, proportional to wind component
                          toward each neighbour.

Temperature uses net flux (antisymmetric) — conserved by construction.

Conservation :
  Excess mist above 7.0 precipitates to ground_water immediately.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

_NEIGHBOURS = [
    (-1,  0),   # north   — wy < 0
    ( 1,  0),   # south   — wy > 0
    ( 0,  1),   # east    — wx > 0
    ( 0, -1),   # west    — wx < 0
    (-1,  1),   # NE
    (-1, -1),   # NW
    ( 1,  1),   # SE
    ( 1, -1),   # SW
]


def step(world: "World") -> None:
    cfg = world.config["atmosphere"]
    mist_to_groundwater = world.config["water"]["mist_to_groundwater"]

    transport_rate  = cfg["wind_transport_rate"]
    diffusion_rate  = cfg.get("mist_diffusion_rate",  0.02)
    advection_rate  = cfg.get("mist_advection_rate",  0.05)

    f  = world.front
    b  = world.back
    wx = f.wind_x   # eastward  (positive col)
    wy = f.wind_y   # southward (positive row)

    # --- Wind component toward each neighbour ---
    # For direction (dr, dc) : component = wx*dc + wy*dr
    # Clamped to [0, inf) — only outflow counts
    wind_toward = []
    for dr, dc in _NEIGHBOURS:
        component = np.clip(
            wx * dc + wy * dr, 0.0, None
        ).astype(np.float32)
        wind_toward.append(component)

    total_wind_raw = np.sum(wind_toward, axis=0).astype(np.float32)
    total_wind     = np.minimum(total_wind_raw, 1.0).astype(np.float32)

    fractions = []
    for wc in wind_toward:
        fractions.append(
            np.where(total_wind_raw > 0.0,
                     wc / (total_wind_raw + 1e-8),
                     0.0).astype(np.float32)
        )

    # --- Temperature transport via net flux (conserved) ---
    new_atmo_temp = f.atmo_temp.copy()
    for i, (dr, dc) in enumerate(_NEIGHBOURS):
        nb_temp  = np.roll(np.roll(f.atmo_temp, -dr, axis=0), -dc, axis=1)
        net_flux = (transport_rate * fractions[i]
                    * (f.atmo_temp - nb_temp)).astype(np.float32)
        new_atmo_temp -= net_flux
        new_atmo_temp += np.roll(np.roll(net_flux,  dr, axis=0),  dc, axis=1)

    # --- Mist transport ---

    # 1. Isotropic diffusion (symmetric, stable)
    new_mist = f.mist.copy()
    n = len(_NEIGHBOURS)
    for dr, dc in _NEIGHBOURS:
        nb_mist   = np.roll(np.roll(f.mist, dr, axis=0), dc, axis=1)
        diff_flux = (diffusion_rate / n * (f.mist - nb_mist)).astype(np.float32)
        new_mist -= diff_flux
        new_mist += np.roll(np.roll(diff_flux, -dr, axis=0), -dc, axis=1)

    # 2. Wind advection (directional)
    mist_outflow = (advection_rate * total_wind * f.mist).astype(np.float32)
    mist_outflow  = np.minimum(mist_outflow, f.mist)
    new_mist     -= mist_outflow
    for i, (dr, dc) in enumerate(_NEIGHBOURS):
        new_mist += np.roll(np.roll(
            (mist_outflow * fractions[i]).astype(np.float32),
            dr, axis=0), dc, axis=1)    # ← +dr, +dc

    # --- Excess mist → ground water ---
    excess_mist      = np.maximum(new_mist - 7.0, 0.0).astype(np.float32)
    new_mist         = np.minimum(new_mist, 7.0).astype(np.float32)
    new_ground_water = (f.ground_water + excess_mist * mist_to_groundwater).astype(np.float32)

    # --- Apply ---
    b.atmo_temp    = new_atmo_temp.astype(np.float32)
    b.mist         = np.maximum(new_mist, 0.0).astype(np.float32)
    b.ground_water = new_ground_water
