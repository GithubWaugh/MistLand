"""
main.py
Entry point for MistLand.
Loads config, generates world, launches the Tkinter/pygame main window.
"""

import json
import time

from sim.world import World
from sim.generation import generate


CONFIG_PATH = "config/default.json"
SEED        = 42


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=== MistLand ===")
    config = load_config(CONFIG_PATH)

    w = config["world"]["grid_width"]
    h = config["world"]["grid_height"]
    print(f"Creating world ({w} × {h})…")
    world = World(config)

    print(f"Generating world (seed={SEED})…")
    t0 = time.perf_counter()
    generate(world, seed=SEED)
    print(f"Generation done in {time.perf_counter() - t0:.3f}s")

    from ui.main_window import MainWindow
    app = MainWindow(config)
    app.start(world)


if __name__ == "__main__":
    main()
