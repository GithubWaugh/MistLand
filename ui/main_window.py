"""
main_window.py
Tkinter main window for MistLand.

Embeds pygame inside a Tk frame using SDL_WINDOWID.
The simulation tick and rendering loop run via root.after().
The native OS menu bar provides File / New Sim / Adjust Params / Help.
"""

import os
import time
import tkinter as tk
from tkinter import messagebox

import pygame

from sim.world import World
from sim.generation import generate
from ui.app import Renderer, WINDOW_W, WINDOW_H, INFO_BAR_H, SPEED_STEPS


class MainWindow:
    """
    Main application window.
    Call start(world, config) to launch.
    """

    def __init__(self, config: dict):
        self.config   = config
        self.world    = None
        self.renderer = None

        self._tick_accum = 0.0
        self._last_time  = 0.0

        self.root = tk.Tk()
        self.root.title("MistLand")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        #self.root.bind("<Key>", self._on_key)
        #self.embed.bind("<Key>", self._on_key)

        self._setup_pygame()
        self._setup_menu()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_key(self, event) :
        """Redirect Tkinter key events to pygame via a synthetic event."""
        # Map Tkinter keysym to pygame key constant
        keymap = {
            "space": pygame.K_SPACE,
            "a": pygame.K_a, "q": pygame.K_q, "i": pygame.K_i,
            # Chiffres avec et sans shift (AZERTY)
            "1": pygame.K_1, "ampersand":    pygame.K_1,   # &
            "2": pygame.K_2, "eacute":       pygame.K_2,   # é
            "3": pygame.K_3, "quotedbl":     pygame.K_3,   # "
            "4": pygame.K_4, "apostrophe":   pygame.K_4,   # '
            "5": pygame.K_5, "parenleft":    pygame.K_5,   # (
            "6": pygame.K_6, "minus":        pygame.K_6,   # -  ← attention, aussi signe moins
            "7": pygame.K_7, "egrave":       pygame.K_7,   # è
            "8": pygame.K_8, "underscore":   pygame.K_8,   # _
            "Prior":  pygame.K_PAGEUP,
            "Next":   pygame.K_PAGEDOWN,
            "Escape": pygame.K_ESCAPE,
        }
        key = keymap.get(event.keysym)
        if key is not None:
            pygame.event.post(pygame.event.Event(
                pygame.KEYDOWN, key=key, mod=0, unicode=""))
        return "break"  # prevent Tk from also handling the event

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_menu(self) -> None:
        menubar = tk.Menu(self.root)

        # File
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save…",  command=self._on_save)
        file_menu.add_command(label="Load…",  command=self._on_load)
        file_menu.add_separator()
        file_menu.add_command(label="Quit",   command=self._on_quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # New Sim
        menubar.add_command(label="New Sim", command=self._on_new_sim)

        # Adjust Params
        menubar.add_command(label="Adjust Params", command=self._on_adjust_params)

        # Help
        menubar.add_command(label="Help", command=self._on_help)

        self.root.config(menu=menubar)

        # Key bindings — must be after embed exists
        self.root.bind_all("<Key>", self._on_key)
        self.embed.bind("<Key>", self._on_key)
        self.embed.focus_set()
        self.embed.bind("<Button-1>", lambda e: self.embed.focus_set())

    def _setup_pygame(self) -> None:
        """Embed pygame into a Tkinter frame using SDL_WINDOWID."""
        w = WINDOW_W
        h = WINDOW_H + INFO_BAR_H

        # Create embed frame — must exist before pygame init
        self.embed = tk.Frame(self.root, width=w, height=h)
        self.embed.pack(fill="both", expand=True)
        self.embed.focus_set()
        self.embed.bind("<Button-1>", lambda e: self.embed.focus_set())
        self.embed.update()   # ensure frame has a real HWND

        # Point SDL at the Tk frame
        os.environ["SDL_VIDEODRIVER"] = "windib"
        os.environ["SDL_WINDOWID"]    = str(self.embed.winfo_id())

        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((w, h))

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self, world: World) -> None:
        """Attach a world and begin the main loop."""
        self.world    = world
        self.renderer = Renderer(world)
        self.renderer.init_fonts()
        self._last_time  = time.perf_counter()
        self._tick_accum = 0.0
        self.root.after(16, self._loop)
        self.root.mainloop()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        if self.world is None:
            self.root.after(16, self._loop)
            return

        now = time.perf_counter()
        dt  = min(now - self._last_time, 0.1)   # cap dt to avoid spiral
        self._last_time = now

        state = self.renderer.state

        # Simulation tick
        if not state.paused:
            self._tick_accum += dt
            interval = 1.0 / SPEED_STEPS[state.speed_idx]
            while self._tick_accum >= interval:
                self.world.tick()
                self._tick_accum -= interval

        # Pygame events
        for event in pygame.event.get():
            actions = self.renderer.handle_event(
                event, self.world, self.screen.get_size())
            if "quit" in actions:
                self._on_quit()
                return
            if "step" in actions:
                self.world.tick()
            if "toggle_pause" in actions:
                state.paused = not state.paused
                self._tick_accum = 0.0
            if "speed_up" in actions:
                state.speed_idx = min(state.speed_idx + 1, len(SPEED_STEPS) - 1)
                self._tick_accum = 0.0
            if "speed_down" in actions:
                state.speed_idx = max(state.speed_idx - 1, 0)
                self._tick_accum = 0.0

        # Render
        self.renderer.render(self.screen, self.world)

        self.root.after(16, self._loop)

    # ------------------------------------------------------------------
    # Menu callbacks
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        if self.world is None:
            return
        from ui.dialogs import save_dialog
        save_dialog(self.root, self.world, on_save=None)

    def _on_load(self) -> None:
        def _apply(world: World):
            self.world    = world
            self.config   = world.config
            self.renderer = Renderer(world)
            self.renderer.init_fonts()
            self._tick_accum = 0.0

        from ui.dialogs import load_dialog
        load_dialog(self.root, on_load=_apply)

    def _on_new_sim(self) -> None:
        def _start(new_config: dict, seed: int):
            world = World(new_config)
            generate(world, seed=seed)
            self.config   = new_config
            self.world    = world
            self.renderer = Renderer(world)
            self.renderer.init_fonts()
            self._tick_accum = 0.0

        from ui.dialogs import new_sim_dialog
        new_sim_dialog(self.root, self.config, on_confirm=_start)

    def _on_adjust_params(self) -> None:
        from ui.dialogs import adjust_params_dialog
        adjust_params_dialog(self.root, self.config)

    def _on_help(self) -> None:
        from ui.dialogs import help_dialog
        help_dialog(self.root)

    def _on_quit(self) -> None:
        pygame.quit()
        self.root.destroy()
