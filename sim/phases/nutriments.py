"""
nutriments.py
Phase 4 : Nutriment diffusion to neighbours.

Each tick, a fraction (diffusion_rate) of each cell's nutriments
diffuses equally to its 4 direct neighbours.
Simulates microfauna movement (insects, worms, etc.)

Nutriments are uint8 [0..255]. Diffusion uses integer arithmetic :
  - Each cell sends floor(diffusion_rate * nutriments) to each neighbour.
  - The actual amount sent = units_per_neighbour * 4 (total outgoing).
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
    (0,  1),   # north
    (0, -1),   # south
    (1, -1),   # east
    (1,  1),   # west
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

    # Total outgoing (4 neighbours)
    total_outgoing = (units_per_neighbour.astype(np.uint16) * 4).astype(np.uint16)

    # Clamp : cannot send more than available
    # (safety check — with diffusion_rate <= 0.25 this should never trigger)
    total_outgoing = np.minimum(
        total_outgoing,
        f.nutriments.astype(np.uint16)
    ).astype(np.uint16)

    # Recompute units_per_neighbour after clamp
    units_per_neighbour = (total_outgoing // 4).astype(np.uint8)
    total_outgoing      = (units_per_neighbour.astype(np.uint16) * 4).astype(np.uint16)

    # --- Build new nutriments : subtract outgoing, add incoming ---
    new_nutriments = (
        f.nutriments.astype(np.int16) - total_outgoing.astype(np.int16)
    ).astype(np.int16)

    for axis, shift in _NEIGHBOURS:
        # Each neighbour receives units_per_neighbour from this cell
        # Deposit in neighbour (inverse roll)
        new_nutriments += np.roll(
            units_per_neighbour.astype(np.int16), -shift, axis=axis
        )

    # Clamp to [0, 255] (safety)
    b.nutriments = np.clip(new_nutriments, 0, 255).astype(np.uint8)