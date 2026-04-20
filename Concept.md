# Concept Notes

## Game concept

Simulation of a finite world, with vegetation, simple living entities, and basic meteorologic behaviors.

---

## Time

### Ticks

One tick = one day.

Player can choose :
- **Automatic** time evolution — one tick per second by default ;
- **Step by step** — via a "one step forward" button.

### Simulation phases order (per tick)

| # | Phase | Description |
|---|-------|-------------|
| 1 | **Temperature** | Ground ↔ atmosphere ↔ neighbours radiation |
| 2 | **Pressure** | Derived from atmospheric temperature and altitude |
| 3 | **Vegetation** | Growth / devolution, water and nutriments consumption |
| 4 | **Nutriments** | Diffusion to neighbours |
| 5 | **Evaporation** | Ground water → atmosphere (mist) |
| 6 | **Atmospheric movements** | Wind transports mist and temperature |
| 7 | **Water movements** | Runoff from high ground to lower neighbours |
| 8 | **Rain** | Mist → ground water, based on temperature and humidity |

---

## Game appearance

Windowed or full-screen, with several frames :

- **Top-down view of the map**
  - Base color / bitmap for the ground layer
  - Overlayed icons for the vegetation (with transparency)
  - Global overlays : hygrometry, temperature, wind, pressure, etc.
- **Text frame** — command-line type interaction with the simulation
- **3D view of the torus**
  - Instancing of low-poly models for vegetation and entities (to be defined later)
- **Pop-up windows**
  - Display of specific cell information
  - Parameters settings

---

## World

Toric planet, modeled as a 2D grid (wrap-around on both axes u and v).

### Global parameters

Settings defined when starting a new game :

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_width` | 1024 | Grid width (u axis) |
| `grid_height` | 512 | Grid height (v axis) |
| `total_water` | 2.0 per cell | Total water in the world (conserved) |
| `total_energy` | — | Total energy in the world (controllable) |

**Data** : grids dumped to files when saving game progress.

---

## World cells

Each cell is defined by a set of static properties and layered data.

### Static properties

- **UVs**
  - `vector2`, normalized value of each cell's cartesian coordinates.
  - *(Will be used to vary incoming light)*
- **Altitude**
  - `float`, generated via simplex 2D noise + blur
  - *(Erosion algorithm planned for a later version)*

- **Base type** : `bare` | `sand` | `soil`
  - `soil` = `sand` + at least 25 organic nutriments
  - Each type has constant properties (defined in config) :

| Base type | Thermal inertia | Water retention |
|-----------|----------------|-----------------|
| `bare` | high | none |
| `sand` | low | low |
| `soil` | medium | medium |

---

### Ground layer *(varying)*

| Field | Type | Description |
|-------|------|-------------|
| `ground_water` | `float [0..1]` | Quantity of water present |
| `nutriments` | `int [0..255]` | Quantity of organic nutriments |
| `ground_temperature` | `float` (°C) | Ground-level temperature |
| `albedo` | `float [0..1]` | Fraction of energy reflected ; updated each tick based on base type, water and vegetation |

---

### Atmospheric layer *(varying)*

| Field | Type | Description |
|-------|------|-------------|
| `pressure` | `float` (bar) | Drives wind — see Atmosphere section |
| `mist` | `int [0..7]` | Airborne water quantity |
| `atmo_temperature` | `float` (°C) | Atmospheric temperature |

---

### Vegetation layer *(varying)*

Type : `None` | `Lichens` | `Grass` | `Shrubs` | `Trees`

Each level implies all lower levels are present (e.g. `Shrubs` implies `Grass` and `Lichens`).

---

### Entities layer *(not implemented yet)*

List of entities present on the cell.

---

### Description *(derived)*

A human-readable label derived from base type, ground water and vegetation.
Examples : *Desert, Swamp, Rainforest, Lake, Tundra…*

---

## Varying data — evolution rules

### Ground level

#### Temperature

- Ground temperature partially radiates each tick :
  - to neighbouring cells, proportionally to cell type thermal inertia ;
  - to the atmosphere above.

#### Water

Fixed world total quantity (conserved).

- **Evaporation** : if ground temperature exceeds `evap_temp_threshold` and ground water > 0, a fraction (`evap_rate`) transfers to mist
- **Runoff** : if a cell is higher than its neighbours, ground water flows to them proportionally to altitude delta ; `sand` and `soil` retain a minimum amount (`retention_min`)
- **Flooding** : if `ground_water` exceeds `flooding_threshold` (varies by base type), the cell becomes a *Lake* :
  - generates nutriments for its neighbours each tick ;
  - if water runs off to neighbours, some nutriments travel with it.

#### Nutriments

Each tick, a fraction (`nutriment_diffusion_rate`) of a cell's nutriments diffuses to its direct neighbours.
*(Simulates microfauna movement — insects, worms, etc.)*

| Sources | Sinks |
|---------|-------|
| Vegetation devolution | Vegetation growth |
| Lake proximity | — |

#### Vegetation

Vegetation evolves at most once every `vegetation_growth_period` ticks (not every tick — growth is slow relative to daily simulation steps).

| Condition | Effect |
|-----------|--------|
| `ground_water` ≥ `veg_water_min` AND `temp_min` ≤ temperature ≤ `temp_max` AND `nutriments` ≥ `veg_nutriments_min` | Evolves upward (e.g. Lichens → Grass → Shrubs → Trees) |
| `ground_water` < `veg_water_min` OR temperature out of range | Devolves downward (e.g. Trees → Shrubs) |

Additional rules :
- Growth consumes `growth_cost_water` and `growth_cost_nutriments`, stored within the plant
- **Lichens** : grow from nothing (no nutriments required) ; release nutriments when destroyed
- Vegetation stores water (`water_stored_per_level` per vegetation level)
- Devolution releases nutriments (`devolution_nutriments_release`) to ground and water (`devolution_water_release`) to atmosphere
- Vegetation reduces cell albedo

---

### Atmospheric level

#### Pressure

Atmospheric pressure is the primary driver of wind. Computed each tick :

```
pressure(cell) = P_base
               - k_temp × atmo_temperature(cell)
               + k_alt  × altitude(cell)
```

- **Hot air → low pressure** (air expands and rises)
- **High altitude → low pressure** (less air mass above)

Both effects lower pressure simultaneously for a hot lowland ; they partially cancel for a cold highland — generating varied wind patterns.

A `pressure_damping` coefficient prevents numerical oscillation.

#### Wind

Derived from the pressure gradient — not stored, computed each tick :

```
wind(A → B) = k_wind × (pressure(A) - pressure(B))
```

Each tick, wind transports the cell's atmospheric content (mist and temperature) to neighbouring cells, proportionally to wind strength.

*(Wider scale rules such as vortices and Coriolis effect — deferred)*

#### Rain

If `atmo_temperature` < `rain_temp_threshold` AND `mist` ≥ `rain_humidity_threshold` :
- A fraction (`rain_rate`) of mist transfers to ground water.

#### Temperature
- Atmospheric temperature partially radiates to the ground.

---

### Entities *(not implemented yet)*

- Animals feeding on vegetation and/or each other ;
- Animals can move to an adjacent cell each tick.

---

## Configuration reference

All tunable parameters are stored in a JSON config file. Default values are indicative and subject to balancing.

```json
{
  "world": {
    "grid_width": 1024,
    "grid_height": 512,
    "total_water": 2.0,
    "total_energy": 50000
  },

  "base_types": {
    "bare": { "thermal_inertia": 0.8, "water_retention": 0.0, "flooding_threshold": 0.2, "albedo_base": 0.40 },
    "sand": { "thermal_inertia": 0.3, "water_retention": 0.05, "flooding_threshold": 0.5, "albedo_base": 0.60 },
    "soil": { "thermal_inertia": 0.5, "water_retention": 0.15, "flooding_threshold": 0.8, "albedo_base": 0.30 }
  },

  "temperature": {
    "ground_to_neighbour_rate": 0.05,
    "ground_to_atmosphere_rate": 0.10,
    "atmosphere_to_ground_rate": 0.08
  },

  "atmosphere": {
    "P_base": 1.0,
    "k_temp": 0.3,
    "k_alt": 0.2,
    "k_wind": 0.5,
    "pressure_damping": 0.1,
    "wind_transport_rate": 0.4
  },

  "water": {
    "evap_temp_threshold": 25.0,
    "evap_rate": 0.05,
    "runoff_rate": 0.3
  },

  "rain": {
    "rain_temp_threshold": 5.0,
    "rain_humidity_threshold": 5,
    "rain_rate": 0.6
  },

  "nutriments": {
    "diffusion_rate": 0.02,
    "lake_generation_rate": 3,
    "runoff_transport_rate": 0.1
  },

  "vegetation": {
    "growth_period_ticks": 30,
    "water_min": 0.2,
    "temp_min": 0.0,
    "temp_max": 40.0,
    "nutriments_min": 10,
    "growth_cost_water": 0.1,
    "growth_cost_nutriments": 5,
    "water_stored_per_level": 0.05,
    "devolution_nutriments_release": 8,
    "devolution_water_release": 0.05
  },

  "albedo": {
    "vegetation_reduction_per_level": 0.05
  }
}
```

---

## Future / deferred features

- **O₂ / CO₂** : atmospheric gas cycle (photosynthesis, respiration) — when entities are implemented
- **Energy exchange with outside** : solar gain, radiative loss, albedo-driven
- **Erosion** : altitude smoothing over time based on water flow
- **Large-scale atmospheric dynamics** : vortices, Coriolis effect
- **Geological events** : volcanoes, particle emissions
