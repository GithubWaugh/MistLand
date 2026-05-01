"""
main.py
Entry point for MistLand.
Loads config, generates world, then launches the pygame UI.
NaN detection active for debugging.
"""

import json
import time
import numpy as np

from sim.world import World
from sim.generation import generate
from sim.phases.evaporation import MIST_UNIT


CONFIG_PATH = "config/default.json"
SEED        = 42

# Set to True to run console-only diagnostics instead of UI
CONSOLE_MODE = False
CONSOLE_TICKS = 130


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def total_water(world: World) -> float:
    return (
        float(world.front.ground_water.sum())
        + float(world.front.mist.sum()) * MIST_UNIT
        + float(world.front.vegetation_water.sum())
    )


def check_nan(world: World) -> bool:
    """
    Check all buffers for NaN values.
    Returns True if any NaN found, and prints which field is affected.
    """
    fields = [
        ("ground_temp",  world.front.ground_temp),
        ("atmo_temp",    world.front.atmo_temp),
        ("pressure",     world.front.pressure),
        ("mist",         world.front.mist),
        ("ground_water", world.front.ground_water),
        ("nutriments",   world.front.nutriments.astype(np.float32)),
        ("albedo",       world.front.albedo),
        ("veg_water",    world.front.vegetation_water),
    ]
    found = False
    for name, arr in fields:
        if np.any(np.isnan(arr)):
            count = int(np.sum(np.isnan(arr)))
            print(f"  *** NaN in {name} : {count} cells affected ***")
            found = True
    return found


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

    if CONSOLE_MODE:
        # Console diagnostics — run N ticks and check for NaN each tick
        print(f"\nRunning {CONSOLE_TICKS} ticks with NaN detection...")
        for _ in range(CONSOLE_TICKS):
            world.tick()
            if check_nan(world):
                print(f"  --> NaN detected at tick {world.tick_count}, stopping.")
                break
            if world.tick_count == 1000:
                print(f"ground_water   : {world.front.ground_water.sum():.4f}")
                print(f"mist           : {world.front.mist.sum() * MIST_UNIT:.4f}")
                print(f"vegetation_water: {world.front.vegetation_water.sum():.4f}")
            if world.tick_count % 10 == 0:
                print(f"Tick {world.tick_count:5d} | "
                      f"water={total_water(world):.4f} | "
                      f"energy={world.total_energy():.4f}")
        print("Done.")

    else:
        # Normal UI mode — NaN check runs silently each tick
        print("Launching UI...")
        from ui.app import run
        run(world)
        print(f"Final : water={total_water(world):.4f}  energy={world.total_energy():.4f}")
        print("Done.")
    
    #print(f"Total atmo  : {world._diag_atmo:+.3f}")
    #print(f"Total water : {world._diag_water:+.3f}")
    #print(f"Total rain  : {world._diag_rain:+.3f}")


if __name__ == "__main__":
    main()