"""
temperature.py
Phase 1 : Temperature propagation.

Each tick :
  - Ground temp exchanges heat with its 4 neighbours (toric wrap),
    weighted by thermal inertia.
  - Ground temp and atmosphere temp exchange heat via a single
    NET FLUX — guaranteeing exact energy conservation.

Conservation fix :
  Instead of two independent fluxes (ground→atmo and atmo→ground),
  we compute a single net flux proportional to the temperature
  DIFFERENCE between ground and atmosphere.
  What leaves one layer enters the other exactly — no energy created.

  net_flux = exchange_rate * (ground_temp - atmo_temp)
  ground loses net_flux, atmosphere gains net_flux.
  If atmo is hotter than ground, net_flux is negative → heat flows down.

Neighbour layout (4-connectivity, toric wrap) :
    north = roll(+1, axis=0)
    south = roll(-1, axis=0)
    east  = roll(-1, axis=1)
    west  = roll(+1, axis=1)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World


_NEIGHBOURS = [
    (0,  1),   # north
    (0, -1),   # south
    (1, -1),   # east
    (1,  1),   # west
]


def step(world: "World") -> None:
    cfg      = world.config["temperature"]
    base_cfg = world.config["base_types"]

    # Single exchange rate for ground ↔ atmosphere
    # Derived from the average of the two former rates
    ga_rate = cfg["ground_to_atmosphere_rate"]
    ag_rate = cfg["atmosphere_to_ground_rate"]
    exchange_rate = (ga_rate + ag_rate) / 2.0

    gn_rate = cfg["ground_to_neighbour_rate"]

    f = world.front
    b = world.back

    # --- Thermal inertia map ---
    inertia = np.empty((world.height, world.width), dtype=np.float32)
    inertia[world.base_type == 0] = base_cfg["bare"]["thermal_inertia"]
    inertia[world.base_type == 1] = base_cfg["sand"]["thermal_inertia"]
    inertia[world.base_type == 2] = base_cfg["soil"]["thermal_inertia"]

    # --- Ground ↔ neighbours (conserved by construction) ---
    outgoing = (gn_rate * inertia * f.ground_temp).astype(np.float32)

    incoming_ground = np.zeros_like(f.ground_temp)
    for axis, shift in _NEIGHBOURS:
        incoming_ground += np.roll(outgoing, shift, axis=axis)

    net_neighbour = incoming_ground - 4.0 * outgoing

    # --- Ground ↔ atmosphere : single net flux ---
    # Positive → heat flows from ground to atmosphere
    # Negative → heat flows from atmosphere to ground
    net_flux = (exchange_rate * (f.ground_temp - f.atmo_temp)).astype(np.float32)

    # --- Update ---
    b.ground_temp = (f.ground_temp + net_neighbour - net_flux).astype(np.float32)
    b.atmo_temp   = (f.atmo_temp   + net_flux).astype(np.float32)