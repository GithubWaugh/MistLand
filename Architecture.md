# Architecture Notes

## Project overview

Python-based world simulation. The backend computes the world state tick by tick in RAM using numpy arrays. The frontend (pygame) reads the current state and renders it. Both run in the same Python process.

---

## Repository structure

```
MistLand/
│
├── config/
│   └── default.json          # All tunable parameters with _comment annotations
│
├── sim/                      # Backend — simulation logic
│   ├── __init__.py
│   ├── world.py              # World class : holds all numpy arrays, exposes tick()
│   ├── buffers.py            # WorldBuffers dataclass (all dynamic arrays + copy_from)
│   ├── generation.py         # World generation : value noise, fertility, initial state
│   ├── io.py                 # Save / load world state (npz + JSON)
│   └── phases/               # One module per simulation phase
│       ├── __init__.py
│       ├── rain.py           # Phase 1 : mist → ground water
│       ├── temperature.py    # Phase 2 : ground ↔ atmosphere heat exchange
│       ├── pressure.py       # Phase 3 : pressure from temperature and surface altitude
│       ├── wind.py           # Phase 4 : Navier-Stokes wind vector field
│       ├── vegetation.py     # Phase 5 : growth / devolution, fertility modulation
│       ├── nutriments.py     # Phase 6 : nutrient diffusion to neighbours
│       ├── evaporation.py    # Phase 7 : ground water → mist
│       ├── atmosphere.py     # Phase 8 : wind transport of mist and temperature
│       └── water.py          # Phase 9 : runoff and flooding
│
├── ui/                       # Frontend — rendering and input
│   ├── __init__.py
│   ├── app.py                # Renderer class : pygame canvas, multi-layer rendering
│   ├── main_window.py        # Tkinter main window embedding pygame, event loop, menu
│   └── dialogs.py            # Modal dialogs : save / load / new sim / adjust params / help
│
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```

---

## Data structures

### World state — numpy arrays

All world data is stored as numpy arrays of shape `(height, width)`, i.e. `[y, x]`.
All arrays live in a single `World` object.

#### Static arrays *(set at generation, never modified during simulation)*

| Array | dtype | Description |
|-------|-------|-------------|
| `altitude` | `float32` | Cell altitude [0..1], generated via seamlessly-tiling value noise (6 octaves) |
| `base_type` | `uint8` | 0 = bare, 1 = sand, 2 = soil |
| `fertility` | `float32` | Local life-favourability [-1..1] ; noise frequency varies by base type |
| `uv` | `float32` | Normalized (u, v) coordinates, shape (height, width, 2) |

#### Ground layer *(updated each tick, stored in WorldBuffers)*

| Array | dtype | Range | Description |
|-------|-------|-------|-------------|
| `ground_water` | `float32` | [0..+∞) | Water at ground level. Above flooding_threshold = lake. |
| `nutriments` | `uint8` | [0..255] | Organic nutriment quantity |
| `ground_temp` | `float32` | °C | Ground temperature |
| `albedo` | `float32` | [0..1] | Energy reflection coefficient |
| `vegetation_water` | `float32` | [0..+∞) | Water stored inside plants (part of conserved total) |

#### Atmospheric layer *(updated each tick)*

| Array | dtype | Range | Description |
|-------|-------|-------|-------------|
| `pressure` | `float32` | bar | Atmospheric pressure (drives wind) |
| `mist` | `float32` | [0..7] | Airborne water — continuous float (no integer rounding artefacts) |
| `atmo_temp` | `float32` | °C | Atmospheric temperature |
| `wind_x` | `float32` | — | East-ward wind component (persistent across ticks) |
| `wind_y` | `float32` | — | South-ward wind component (positive = screen-down) |

#### Vegetation layer *(updated each tick)*

| Array | dtype | Range | Description |
|-------|-------|-------|-------------|
| `vegetation` | `uint8` | [0..4] | 0=None, 1=Lichens, 2=Grass, 3=Shrubs, 4=Trees |
| `veg_tick_counter` | `uint16` | ticks | Ticks since last vegetation change |

---

### Double buffer

The `World` object maintains two `WorldBuffers` instances : **front** and **back**.

- The renderer reads from `front`
- Each phase reads from `front` and writes to `back`
- After each phase : `swap_buffers()` + `sync_back_from_front()` — back becomes the new front for the next phase

```python
# After each phase :
world.front, world.back = world.back, world.front
world.back.copy_from(world.front)   # sync for next phase
```

This pattern ensures every phase sees a fully consistent, up-to-date world state, and inter-phase dependencies are respected (e.g. evaporation updates mist before atmosphere reads it).

---

## Tick loop

```python
def tick(self) -> None:
    self.sync_back_from_front()

    rain.step(self);         self.swap_buffers(); self.sync_back_from_front()  # mist → ground water first
    temperature.step(self);  self.swap_buffers(); self.sync_back_from_front()
    pressure.step(self);     self.swap_buffers(); self.sync_back_from_front()
    wind.step(self);         self.swap_buffers(); self.sync_back_from_front()
    vegetation.step(self);   self.swap_buffers(); self.sync_back_from_front()
    nutriments.step(self);   self.swap_buffers(); self.sync_back_from_front()
    evaporation.step(self);  self.swap_buffers(); self.sync_back_from_front()
    atmosphere.step(self);   self.swap_buffers(); self.sync_back_from_front()
    water.step(self);        self.swap_buffers(); self.sync_back_from_front()

    self.tick_count += 1
```

Each phase module exposes a single `step(world)` function that reads from `world.front` and writes to `world.back`.

---

## Conservation laws

Two quantities are strictly conserved across all ticks :

**Total water** = `ground_water.sum() + mist.sum() × MIST_UNIT + vegetation_water.sum()`
- `mist` is `float32` to avoid integer rounding losses
- `water.py` applies a float32 rounding correction after runoff
- `atmosphere.py` converts excess mist > 7.0 to ground water instead of clipping

**Total energy** = `ground_temp.sum() + atmo_temp.sum()`
- `temperature.py` uses a net flux formula : what leaves ground enters atmosphere exactly
- `atmosphere.py` uses antisymmetric net flux for temperature transport

---

## Key implementation notes

### Mist : float32 instead of uint8

`mist` was originally `uint8 [0..7]`. It was changed to `float32 [0..7]` to eliminate integer rounding artefacts (checkerboard pattern in cloud display and water conservation errors).
The `mist_accumulator` field (previously used to buffer fractional evaporation) was removed as a consequence.

### Hydraulic altitude for runoff

Runoff in `water.py` uses hydraulic altitude rather than terrain altitude. Only water *above* the flooding threshold is counted, so shallow water doesn't inflate the surface height unduly :
```
surface_altitude = altitude + max(ground_water − flooding_threshold, 0) × water_to_altitude
```
This prevents water from flowing into already-flooded cells that happen to be at lower terrain altitude. The same `surface_altitude()` helper is also used by the pressure phase.

### Wind — persistent Navier-Stokes field

`wind_x` and `wind_y` are persistent arrays updated each tick by `wind.py`. Three forces act :

1. **Pressure gradient** : `wind -= k_wind × ∇P` (central differences, toric wrap)
2. **Advection** : `wind -= advection_strength × (wind·∇)wind` (creates vortices)
3. **Viscosity** : `wind += viscosity × ∇²wind` (4-neighbour Laplacian, smooths instabilities)

After the update, a multiplicative `wind_damping` factor and a `wind_max_speed` clamp prevent unlimited acceleration and numerical divergence.

### Seamlessly-tiling noise

`generation.py` uses value noise with `period = frequency` (integer) for each octave. This guarantees `noise(x=0) == noise(x=frequency)` on both axes — perfectly seamless on the torus.

### Vegetation desynchronisation

When a cell grows or devolves, its `veg_tick_counter` resets to a random value in `[0, growth_period/2]` rather than 0. This breaks the synchronous wave behaviour seen in large homogeneous patches.

---

## Threading model

Currently single-threaded : the simulation tick runs in the pygame main loop (accumulator-based timing). No separate thread is needed since tick times are well under 16ms at current grid sizes.

```python
# In pygame loop :
tick_accum += dt
interval = 1.0 / ticks_per_second
while tick_accum >= interval:
    world.tick()
    tick_accum -= interval
```

*(Multi-threading planned if larger grids require it)*

---

## World generation

`generation.py` populates the world in order :

1. **Altitude** : 6-octave value noise (freqs 2, 4, 8, 16, 32, 64) + gaussian blur
2. **Base type** : altitude thresholds (bare < 0.3, sand < 0.6, soil ≥ 0.6)
3. **Fertility** : per-base-type noise at different frequencies, blended into a [-1..1] field
4. **Water distribution** : weighted by inverse altitude (low areas get more water)
5. **Temperature** : latitudinal gradient `cos(π·v)` + altitude lapse rate
6. **Albedo** : from base type

---

## UI

The frontend is split across three files :

- **`main_window.py`** — Tkinter main window. Embeds the Pygame surface via `SDL_WINDOWID`. Drives the event loop with `root.after(16, ...)`. Owns the menu bar (File, New Sim, Adjust Params, Help) and accumulator-based tick timing.
- **`app.py`** — `Renderer` class. All Pygame drawing : hillshading, base layer, vegetation icons, mist veil, wind streamers, rain particles, inspect panel, HSL compositing.
- **`dialogs.py`** — Modal Tkinter dialogs for save, load, new sim, parameter adjustment, and help.

### Rendering layer stack (bottom → top)

1. Hillshading from altitude gradient (always visible)
2. Exclusive data layer (L1) : water level, temperature, altitude spectral map, or plain soil colors
3. Ground elements layer (L2) : vegetation icons or nutriment dots
4. Mist overlay : white veil, per-pixel alpha
5. Wind streamers : directional arrows
6. Rain particle effect
7. Inspect panel : cell info following mouse cursor

HSL compositing : hue + saturation come from the data layer ; luminosity comes from hillshading.

### Controls

| Feature | Key | Notes |
|---------|-----|-------|
| Step one tick | Space | Only when paused |
| Pause / resume | A | |
| Speed control | PgUp / PgDn | 8 steps : 0.25 to 50 t/s |
| Water overlay | 1 | Exclusive with other L1 overlays |
| Temperature overlay | 2 | Exclusive with other L1 overlays |
| Wind streamers | 3 | Toggle |
| Rain effect | 4 | Toggle |
| Altitude overlay | 5 | Spectral (violet→red), exclusive |
| Mist overlay | 6 | White veil, per-pixel alpha |
| Vegetation icons | V | Toggle ; density zoom-adaptive |
| Inspect panel | I | Cell info following mouse cursor |
| Zoom | Mouse wheel | 1x – 32x, centred on cursor |
| Pan | RMB / MMB drag | Clamped to world bounds |

Lakes are displayed in blue on the base layer (dynamic, recomputed each frame).
Submerged lichen is displayed as a teal ~ (algae icon).

---

## Save / load

World state is saved as `.npz` archive + metadata JSON, implemented in `sim/io.py` and wired to the File menu in `ui/main_window.py` and `ui/dialogs.py` :

```
saves/
└── save_001/
    ├── meta.json        # tick count, grid size, config snapshot
    └── state.npz        # all WorldBuffers arrays compressed
```

---

## Configuration

`config/default.json` is loaded at startup into a plain Python dict. All phase modules read parameters from this dict — nothing is hardcoded. Keys prefixed `_comment` are documentation and are ignored.

---

## Build milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | **Data structures** | ✅ World class, WorldBuffers, config loading |
| 2 | **World generation** | ✅ Value noise, fertility, temperature gradient |
| 3 | **Tick loop + all 9 phases** | ✅ Validated (water + energy conservation) |
| 4 | **2D map view** | ✅ Tkinter + pygame, zoom, pan, hillshading, lake colors |
| 5 | **Overlays + inspect** | ✅ Water, temp, altitude (spectral), mist, wind streamers, inspect panel |
| 6 | **UI controls** | ✅ Step/auto/speed, all toggle keys, menu bar, dialogs |
| 7 | **Save / load** | ✅ npz archive + metadata JSON (io.py + dialogs wired) |
| 8 | **3D torus view** | ⏳ ModernGL, instanced vegetation models |
| 9 | **Entities** | ⏳ Animals, feeding, movement |

---

## Dependencies

```
numpy          # world state arrays and all vectorized operations
pygame         # 2D rendering, event loop
scipy          # gaussian_filter for noise smoothing
moderngl       # 3D torus view (milestone 8, not yet used)
opensimplex    # simplex noise (available, alternative to value noise)
```

```
pip install -r requirements.txt
```