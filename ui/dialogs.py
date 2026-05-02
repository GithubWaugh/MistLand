"""
dialogs.py
Tkinter dialog windows for MistLand.

  SaveDialog        — choose a directory, save world state
  LoadDialog        — choose a directory, load world state
  NewSimDialog      — configure and start a new simulation
  AdjustParamsDialog — tweak live simulation parameters
  HelpDialog        — keyboard shortcuts reference
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_dialog(parent, world, on_save) -> None:
    """Open a directory chooser and save the world."""
    from sim import io as sim_io

    path = filedialog.askdirectory(
        parent=parent,
        title="Save simulation — choose or create a folder",
    )
    if not path:
        return
    try:
        sim_io.save(world, path)
        messagebox.showinfo("Saved",
                            f"Simulation saved to:\n{path}",
                            parent=parent)
        if on_save:
            on_save(path)
    except Exception as e:
        messagebox.showerror("Save failed", str(e), parent=parent)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_dialog(parent, on_load) -> None:
    """Open a directory chooser and load a world."""
    from sim import io as sim_io

    path = filedialog.askdirectory(
        parent=parent,
        title="Load simulation — choose a saved folder",
    )
    if not path:
        return
    meta_path = Path(path) / "meta.json"
    if not meta_path.exists():
        messagebox.showerror("Load failed",
                             "No valid save found in this folder.\n"
                             "(meta.json missing)",
                             parent=parent)
        return
    try:
        world = sim_io.load(path)
        if on_load:
            on_load(world)
    except Exception as e:
        messagebox.showerror("Load failed", str(e), parent=parent)


# ---------------------------------------------------------------------------
# New Sim
# ---------------------------------------------------------------------------

def new_sim_dialog(parent, default_config: dict, on_confirm) -> None:
    """
    Dialog to configure a new simulation.
    on_confirm(config, seed) is called if the user confirms.
    """
    win = tk.Toplevel(parent)
    win.title("New Simulation")
    win.resizable(False, False)
    win.grab_set()

    pad = {"padx": 8, "pady": 4}

    fields = [
        ("Grid width",    "grid_width",    str(default_config["world"]["grid_width"])),
        ("Grid height",   "grid_height",   str(default_config["world"]["grid_height"])),
        ("Total water",   "total_water",   str(default_config["world"]["total_water"])),
        ("Random seed",   "seed",          "42"),
    ]

    entries = {}
    for i, (label, key, default) in enumerate(fields):
        tk.Label(win, text=label).grid(row=i, column=0, sticky="e", **pad)
        e = ttk.Entry(win, width=12)
        e.insert(0, default)
        e.grid(row=i, column=1, sticky="w", **pad)
        entries[key] = e

    def _confirm():
        try:
            new_config = {k: v for k, v in default_config.items()}
            new_config["world"] = dict(default_config["world"])
            new_config["world"]["grid_width"]  = int(entries["grid_width"].get())
            new_config["world"]["grid_height"] = int(entries["grid_height"].get())
            new_config["world"]["total_water"] = float(entries["total_water"].get())
            seed = int(entries["seed"].get())
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e), parent=win)
            return

        if not messagebox.askyesno(
            "Confirm",
            "The current simulation will be replaced.\nContinue?",
            parent=win
        ):
            return

        win.destroy()
        if on_confirm:
            on_confirm(new_config, seed)

    btn_frame = tk.Frame(win)
    btn_frame.grid(row=len(fields), column=0, columnspan=2, pady=8)
    ttk.Button(btn_frame, text="Start",  command=_confirm).pack(side="left",  padx=4)
    ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side="left", padx=4)


# ---------------------------------------------------------------------------
# Adjust params
# ---------------------------------------------------------------------------

_PARAMS = [
    # (section, key, label, min, max, is_float)
    # Water
    ("water",      "evap_temp_threshold", "Evap temp threshold (°C)",  0.0,  60.0, True),
    ("water",      "evap_rate",           "Evaporation rate",           0.0,   0.5, True),
    ("water",      "runoff_rate",         "Runoff rate",                0.0,   0.5, True),
    ("water",      "water_to_altitude",   "Water-to-altitude factor",   0.0,   2.0, True),
    # Rain
    ("rain",       "rain_temp_threshold", "Rain temp threshold (°C)",  -10.0, 60.0, True),
    ("rain",       "rain_humidity_threshold", "Rain humidity threshold", 0.0,  7.0, True),
    ("rain",       "rain_rate",           "Rain rate",                  0.0,   1.0, True),
    # Vegetation
    ("vegetation", "growth_period_ticks", "Growth period (ticks)",      1.0, 500.0, False),
    ("vegetation", "water_min",           "Vegetation water min",       0.0,   0.5, True),
    ("vegetation", "temp_min",            "Vegetation temp min (°C)",  -20.0, 30.0, True),
    ("vegetation", "temp_max",            "Vegetation temp max (°C)",   20.0, 80.0, True),
    ("vegetation", "mist_water_threshold", "Mist water threshold",      0.0,  10.0, True),
    # Atmosphere
    ("atmosphere", "k_wind",              "Wind strength (k_wind)",     0.0,   2.0, True),
    ("atmosphere", "wind_transport_rate", "Wind transport rate",        0.0,   1.0, True),
    ("atmosphere", "mist_diffusion_rate", "Mist diffusion rate",        0.0,   0.2, True),
    ("atmosphere", "mist_advection_rate", "Mist advection rate",        0.0,   0.3, True),
    ("world",      "solar_radiation",     "Solar radiation",            0.0,   5.0, True),
]


def adjust_params_dialog(parent, config: dict) -> None:
    """
    Live parameter editor. Changes apply immediately to `config`.
    """
    win = tk.Toplevel(parent)
    win.title("Adjust Parameters")
    win.resizable(False, True)
    win.grab_set()

    canvas = tk.Canvas(win, width=480, height=420)
    sb     = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    pad = {"padx": 6, "pady": 3}

    def _make_row(parent, row_idx, section, key, label, mn, mx, is_float):
        current = config[section].get(key, 0)
        tk.Label(parent, text=label, anchor="w", width=34).grid(
            row=row_idx, column=0, sticky="w", **pad)
        var = tk.StringVar(value=str(current))
        e   = ttk.Entry(parent, textvariable=var, width=10)
        e.grid(row=row_idx, column=1, **pad)

        def _apply(*_):
            try:
                v = float(var.get()) if is_float else int(float(var.get()))
                v = max(mn, min(mx, v))
                config[section][key] = v
            except ValueError:
                pass

        e.bind("<Return>",   _apply)
        e.bind("<FocusOut>", _apply)

    for i, (sec, key, lbl, mn, mx, is_float) in enumerate(_PARAMS):
        _make_row(inner, i, sec, key, lbl, mn, mx, is_float)

    ttk.Button(win, text="Close", command=win.destroy).pack(pady=8)


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

def help_dialog(parent) -> None:
    """Show keyboard shortcuts."""
    win = tk.Toplevel(parent)
    win.title("MistLand — Keyboard Shortcuts")
    win.resizable(False, False)

    shortcuts = [
        ("SPACE",        "Step one tick (when paused)"),
        ("A",            "Pause / Resume"),
        ("Page Up",      "Increase simulation speed"),
        ("Page Down",    "Decrease simulation speed"),
        ("",             ""),
        ("1",            "Toggle water overlay"),
        ("2",            "Toggle temperature overlay"),
        ("3",            "Toggle pressure overlay"),
        ("4",            "Toggle vegetation icons"),
        ("5",            "Toggle altitude overlay (spectral)"),
        ("6",            "Toggle mist / cloud overlay"),
        ("7",            "Toggle wind streamers"),
        ("8",            "Toggle rain overlay"),
        ("9",            "Toggle fertility overlay"),
        ("I",            "Toggle inspect panel (follows cursor)"),
        ("",             ""),
        ("Mouse wheel",  "Zoom in / out (centred on cursor)"),
        ("Right-click drag", "Pan the map"),
        ("Middle-click drag","Pan the map"),
        ("",             ""),
        ("ESC / Q",      "Quit"),
    ]

    frame = tk.Frame(win, padx=16, pady=12)
    frame.pack()
    for key, desc in shortcuts:
        if not key and not desc:
            tk.Label(frame, text="").pack(anchor="w")
            continue
        row = tk.Frame(frame)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=key,  width=22, anchor="w",
                 font=("Courier", 10, "bold")).pack(side="left")
        tk.Label(row, text=desc, anchor="w").pack(side="left")

    ttk.Button(win, text="Close", command=win.destroy).pack(pady=8)
