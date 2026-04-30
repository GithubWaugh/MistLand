"""
app.py
Milestone 4 : pygame window — 2D top-down map view.
Resizable window with zoom and pan.

Displays the world grid as a colored surface :
  - bare  → dark grey
  - sand  → sandy beige
  - soil  → warm brown

Controls :
  - SPACE           : step one tick forward
  - A               : toggle auto mode (one tick per second)
  - Mouse wheel     : zoom in / out
  - Right-click drag or middle-click drag : pan
  - ESC / Q         : quit
"""

import pygame
import numpy as np
import time

from sim.world import World


# --- Colours (R, G, B) ---
COLOR_BARE  = (90,  85,  80)
COLOR_SAND  = (210, 185, 130)
COLOR_SOIL  = (120,  85,  50)
COLOR_BG    = (20,  20,  20)

# UI
WINDOW_TITLE    = "MistLand"
WINDOW_W        = 1024
WINDOW_H        = 512
INFO_BAR_H      = 24
TICK_INTERVAL   = 1.0       # seconds between ticks in auto mode
ZOOM_MIN        = 1
ZOOM_MAX        = 32
ZOOM_DEFAULT    = 4
INFO_FONT_SIZE  = 14


def _build_world_rgb(world: World) -> np.ndarray:
    """
    Return an (height, width, 3) uint8 RGB array from base_type.
    Static — only needs to be built once.
    """
    h, w = world.height, world.width
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[world.base_type == 0] = COLOR_BARE
    rgb[world.base_type == 1] = COLOR_SAND
    rgb[world.base_type == 2] = COLOR_SOIL
    return rgb


def run(world: World) -> None:
    """
    Main pygame loop.
    Call this after world generation is complete.
    """
    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_W, WINDOW_H + INFO_BAR_H),
        pygame.RESIZABLE
    )
    pygame.display.set_caption(WINDOW_TITLE)
    clock   = pygame.time.Clock()
    font    = pygame.font.SysFont("monospace", INFO_FONT_SIZE)

    # --- World RGB array (static, base type only for now) ---
    world_rgb = _build_world_rgb(world)   # (h, w, 3)
    world_w   = world.width
    world_h   = world.height

    # --- Camera state ---
    zoom     = ZOOM_DEFAULT          # pixels per cell
    cam_x    = 0.0                   # world-space X of top-left corner (float)
    cam_y    = 0.0                   # world-space Y of top-left corner (float)

    # --- Pan state ---
    panning      = False
    pan_start_mx = 0     # mouse X when pan started
    pan_start_my = 0     # mouse Y when pan started
    pan_start_cx = 0.0   # cam_x when pan started
    pan_start_cy = 0.0   # cam_y when pan started

    # --- Sim state ---
    auto_mode      = False
    last_tick_time = time.perf_counter()
    running        = True

    def clamp_camera():
        """Keep camera within world bounds."""
        nonlocal cam_x, cam_y
        sw, sh = screen.get_size()
        view_w = (sw) / zoom
        view_h = (sh - INFO_BAR_H) / zoom
        cam_x = max(0.0, min(cam_x, world_w - view_w))
        cam_y = max(0.0, min(cam_y, world_h - view_h))

    def render_map():
        """
        Blit the visible portion of the world onto the screen.
        Crops the world_rgb array to the visible region, scales it by zoom,
        and blits it at (0, 0).
        """
        sw, sh = screen.get_size()
        view_w_px = sw
        view_h_px = sh - INFO_BAR_H

        # Cell range visible
        cell_x0 = int(cam_x)
        cell_y0 = int(cam_y)
        cells_x  = -(-view_w_px // zoom) + 1   # ceil division + 1 for partial cells
        cells_y  = -(-view_h_px // zoom) + 1

        cell_x1 = min(cell_x0 + cells_x, world_w)
        cell_y1 = min(cell_y0 + cells_y, world_h)

        # Crop RGB array to visible region
        crop = world_rgb[cell_y0:cell_y1, cell_x0:cell_x1]   # (rows, cols, 3)

        if crop.size == 0:
            return

        # Build surface from crop
        # pygame.surfarray expects (width, height, 3)
        crop_surf = pygame.surfarray.make_surface(crop.transpose(1, 0, 2))

        # Scale to zoom
        scaled_w = (cell_x1 - cell_x0) * zoom
        scaled_h = (cell_y1 - cell_y0) * zoom
        scaled_surf = pygame.transform.scale(crop_surf, (scaled_w, scaled_h))

        # Offset within the first cell (sub-cell pan)
        offset_x = -int((cam_x - cell_x0) * zoom)
        offset_y = -int((cam_y - cell_y0) * zoom)

        screen.fill(COLOR_BG, (0, 0, sw, view_h_px))
        screen.blit(scaled_surf, (offset_x, offset_y))

    def render_info():
        """Render the info bar at the bottom of the screen."""
        sw, sh = screen.get_size()
        info_rect = pygame.Rect(0, sh - INFO_BAR_H, sw, INFO_BAR_H)
        pygame.draw.rect(screen, (20, 20, 20), info_rect)
        mode_str = "AUTO" if auto_mode else "STEP"
        info_str = (
            f"Tick {world.tick_count:5d}  |  "
            f"Zoom {zoom:2d}x  |  "
            f"[SPACE] step  [A] auto ({mode_str})  "
            f"[Wheel] zoom  [RMB] pan  [ESC] quit"
        )
        text_surf = font.render(info_str, True, (180, 180, 180))
        screen.blit(text_surf, (8, sh - INFO_BAR_H + 4))

    while running:
        sw, sh = screen.get_size()

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    world.tick()
                elif event.key == pygame.K_a:
                    auto_mode = not auto_mode
                    last_tick_time = time.perf_counter()

            elif event.type == pygame.MOUSEWHEEL:
                # Zoom toward mouse position
                mx, my = pygame.mouse.get_pos()
                # World position under mouse before zoom
                world_mx = cam_x + mx / zoom
                world_my = cam_y + my / zoom

                zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom + event.y))

                # Adjust camera so the world position under mouse stays fixed
                cam_x = world_mx - mx / zoom
                cam_y = world_my - my / zoom
                clamp_camera()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (2, 3):   # middle or right click
                    panning      = True
                    pan_start_mx, pan_start_my = event.pos
                    pan_start_cx = cam_x
                    pan_start_cy = cam_y

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (2, 3):
                    panning = False

            elif event.type == pygame.MOUSEMOTION:
                if panning:
                    dx = (event.pos[0] - pan_start_mx) / zoom
                    dy = (event.pos[1] - pan_start_my) / zoom
                    cam_x = pan_start_cx - dx
                    cam_y = pan_start_cy - dy
                    clamp_camera()

            elif event.type == pygame.VIDEORESIZE:
                clamp_camera()

        # --- Auto tick ---
        if auto_mode:
            now = time.perf_counter()
            if now - last_tick_time >= TICK_INTERVAL:
                world.tick()
                last_tick_time = now

        # --- Draw ---
        render_map()
        render_info()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()