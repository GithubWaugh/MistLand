"""
main.py
Entry point for MistLand.
Loads config, generates world, then launches the pygame UI.
"""

import json
import time

from sim.world import World
from sim.generation import generate
from sim.phases.evaporation import MIST_UNIT


CONFIG_PATH = "config/default.json"
SEED        = 42


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def total_water(world: World) -> float:
    return (
        float(world.front.ground_water.sum())
        + float(world.front.mist.sum()) * MIST_UNIT
        + float(world.front.mist_accumulator.sum())
        + float(world.front.vegetation_water.sum())
    )


def main():
    print("=== MistLand ===")
    print(f"Loading config from '{CONFIG_PATH}'...")
    config = load_config(CONFIG_PATH)

    width  = config["world"]["grid_width"]
    height = config["world"]["grid_height"]
    print(f"Creating world ({width} x {height})...")
    world = World(config)

    print(f"Generating world (seed={SEED})...")
    t0 = time.perf_counter()
    generate(world, seed=SEED)
    t1 = time.perf_counter()
    print(f"Generation done in {t1 - t0:.3f}s")
    print(f"Baseline : water={total_water(world):.4f}  energy={world.total_energy():.4f}")

    print("Launching UI...")
    from ui.app import run
    run(world)

    print(f"Final : water={total_water(world):.4f}  energy={world.total_energy():.4f}")
    print("Done.")


if __name__ == "__main__":
    main()