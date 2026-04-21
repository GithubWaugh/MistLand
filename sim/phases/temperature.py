"""
temperature.py
Phase 1 : Temperature propagation.

Each tick :
  - Ground temp radiates a fraction to each of its 4 neighbours,
    weighted by the cell's thermal inertia (from base_type config).
  - Ground temp radiates a fraction to the atmosphere above.
  - Atmosphere temp radiates a fraction to the ground below.

Conservation : total energy (sum of all ground_temp + atmo_temp)
should remain constant across ticks.

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


# Precomputed neighbour shifts — (axis, shift) pairs for np.roll
_NEIGHBOURS = [
    (0,  1),   # north
    (0, -1),   # south
    (1, -1),   # east
    (1,  1),   # west
]


def step(world: "World") -> None:
    cfg      = world.config["temperature"]
    base_cfg = world.config["base_types"]

    ga_rate = cfg["ground_to_atmosphere_rate"]   # ground → atmosphere
    ag_rate = cfg["atmosphere_to_ground_rate"]   # atmosphere → ground
    gn_rate = cfg["ground_to_neighbour_rate"]    # ground → each neighbour

    f = world.front   # read
    b = world.back    # write

    # --- Thermal inertia map (per cell, derived from base_type) ---
    # Higher inertia → cell changes temperature more slowly
    # Used as a multiplier on the outgoing neighbour radiation rate.
    inertia = np.empty((world.height, world.width), dtype=np.float32)
    inertia[world.base_type == 0] = base_cfg["bare"]["thermal_inertia"]
    inertia[world.base_type == 1] = base_cfg["sand"]["thermal_inertia"]
    inertia[world.base_type == 2] = base_cfg["soil"]["thermal_inertia"]

    # --- Ground → atmosphere exchange ---
    # Energy leaving ground, entering atmosphere
    ground_to_atmo = ga_rate * f.ground_temp          # float32 grid
    atmo_to_ground = ag_rate * f.atmo_temp             # float32 grid

    # --- Ground → neighbours radiation ---
    # Each cell sends gn_rate * inertia of its temperature to each neighbour.
    # Total outgoing from ground to neighbours = 4 * gn_rate * inertia * temp
    # Each cell also receives from each of its 4 neighbours.

    # Outgoing flux from each cell to one neighbour
    outgoing = gn_rate * inertia * f.ground_temp       # per-neighbour flux

    # Net ground neighbour exchange : sum of incoming from 4 neighbours - 4 * outgoing
    incoming_ground = np.zeros_like(f.ground_temp)
    for axis, shift in _NEIGHBOURS:
        incoming_ground += np.roll(outgoing, shift, axis=axis)

    net_neighbour = incoming_ground - 4.0 * outgoing

    # --- Update ground temperature ---
    b.ground_temp = (
        f.ground_temp
        + net_neighbour          # neighbour exchange (conserved across grid)
        - ground_to_atmo         # loss to atmosphere
        + atmo_to_ground         # gain from atmosphere
    ).astype(np.float32)

    # --- Update atmosphere temperature ---
    b.atmo_temp = (
        f.atmo_temp
        + ground_to_atmo         # gain from ground
        - atmo_to_ground         # loss to ground
    ).astype(np.float32)

    # Note : atmosphere horizontal transport (wind) is handled in atmosphere.py