"""
main.py
Entry point for MistLand.
For now : loads config, generates world, runs N ticks in console,
and prints conservation law checks after each tick.
"""

import json
import time
import numpy as np

from sim.world import World
from sim.generation import generate


CONFIG_PATH = "config/default.json"
N_TICKS     = 100       # number of ticks to run in this test
SEED        = 42       # world generation seed


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=== MistLand ===")
    print(f"Loading config from '{CONFIG_PATH}'...")
    config = load_config(CONFIG_PATH)

    w = config["world"]["grid_width"]
    h = config["world"]["grid_height"]
    print(f"Creating world ({w} x {h})...")
    world = World(config)

    print(f"Generating world (seed={SEED})...")
    t0 = time.perf_counter()
    generate(world, seed=SEED)
    t1 = time.perf_counter()
    print(f"Generation done in {t1 - t0:.3f}s")

    # Baseline conservation check
    water_0  = world.total_water()
    energy_0 = world.total_energy()
    print(f"\nBaseline : water={water_0:.4f}  energy={energy_0:.4f}")
    print(f"{'Tick':>6}  {'Water':>12}  {'ΔWater':>10}  {'Energy':>12}  {'ΔEnergy':>10}  {'Time':>8}")
    print("-" * 70)

    for _ in range(N_TICKS):
        t0 = time.perf_counter()
        world.tick()
        t1 = time.perf_counter()

        water   = world.total_water()
        energy  = world.total_energy()
        print(
            f"{world.tick_count:>6}  "
            f"{water:>12.4f}  "
            f"{water  - water_0:>+10.6f}  "
            f"{energy:>12.4f}  "
            f"{energy - energy_0:>+10.6f}  "
            f"{(t1-t0)*1000:>6.1f}ms"
            f"Pressure min={world.front.pressure.min():.4f}  "
            f"max={world.front.pressure.max():.4f}  "
            f"mean={world.front.pressure.mean():.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
