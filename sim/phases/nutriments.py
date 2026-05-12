"""
nutriments.py
Phase 4 : Nutriment diffusion to neighbours.

Each tick, a fraction (diffusion_rate) of each cell's nutriments
diffuses equally to its 8 direct neighbours.
Simulates microfauna movement (insects, worms, etc.)

Nutriments are uint8 [0..255]. Diffusion uses integer arithmetic :
  - Each cell sends floor(diffusion_rate * nutriments) to each neighbour.
  - The actual amount sent = units_per_neighbour * 8 (total outgoing).
  - Conservation : what leaves one cell enters its neighbours exactly.

Note : nutriments are also modified by water.py (lakes, runoff)
and vegetation.py (growth, devolution). Those phases handle
their own nutriment changes independently.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World


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
    cfg = world.config["nutriments"]

    diffusion_rate = cfg["diffusion_rate"]

    f = world.front
    b = world.back

    # --- Units sent to each neighbour (integer, floor) ---
    # Using floor ensures we never send more than available.
    units_per_neighbour = np.floor(
        diffusion_rate * f.nutriments.astype(np.float32)
    ).astype(np.uint8)

    # Total outgoing (8 neighbours)
    total_outgoing = (units_per_neighbour.astype(np.uint16) * 8).astype(np.uint16)

    # Clamp : cannot send more than available
    # (safety check — with diffusion_rate <= 0.25 this should never trigger)
    total_outgoing = np.minimum(
        total_outgoing,
        f.nutriments.astype(np.uint16)
    ).astype(np.uint16)

    # Recompute units_per_neighbour after clamp
    units_per_neighbour = (total_outgoing // 8).astype(np.uint8)
    total_outgoing      = (units_per_neighbour.astype(np.uint16) * 8).astype(np.uint16)

    # --- Build new nutriments : subtract outgoing, add incoming ---
    new_nutriments = (
        f.nutriments.astype(np.int16) - total_outgoing.astype(np.int16)
    ).astype(np.int16)

    for dr, dc in _NEIGHBOURS:
        # Each neighbour receives units_per_neighbour from this cell
        # Deposit in neighbour (inverse roll)
        new_nutriments += np.roll(
            np.roll(units_per_neighbour.astype(np.int16), -dr, axis=0), -dc, axis=1
        )
    
    # Banks neighbours of flooded cells receive some free nutriments, to simulate micro-fauna decomposition in lakes. The amount is proportional to the water level (up to 16 units at max water level).
    for dr, dc in _NEIGHBOURS:
        new_nutriments += np.roll(
            np.roll((world.front.ground_water // 32).astype(np.int16), -dr, axis=0), -dc, axis=1
        ) * (world.front.ground_water > 1)  # Only if there's actually water (not just a flooded cell with 1 unit of water)

    # Clamp to [0, 255] (safety); flooded cells (lakes) are capped at 16
    clamped = np.clip(new_nutriments, 0, 255).astype(np.uint8)
    lake_mask = world.get_flooded_mask()
    b.nutriments = clamped * ~(world.front.ground_water > 1) 