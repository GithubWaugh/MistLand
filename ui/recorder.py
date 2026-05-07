"""
recorder.py
Captures pygame frames and encodes them to MP4 using imageio + imageio-ffmpeg.

Recording is active by default on startup.
R key toggles pause / resume — frames already buffered are kept, new ones
are appended after resume, so the final video is one continuous sequence.
On quit, encode() must be called explicitly.
"""
from __future__ import annotations
import os
import tempfile
from typing import Optional

import numpy as np
import pygame

_QUALITY_MAP = {"low": 3, "medium": 6, "high": 9}


class VideoRecorder:
    def __init__(self, config: dict):
        out             = config.get("output", {})
        self.frame_rate = int(out.get("frame_rate", 24))
        self.frame_step = int(out.get("frame_step", 1))
        self._quality   = _QUALITY_MAP.get(str(out.get("quality", "high")), 9)
        self._frames: list[np.ndarray] = []
        self.paused     = False

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def add_frame(self, surface: pygame.Surface, tick: int) -> None:
        if self.paused or tick % self.frame_step != 0:
            return
        arr = pygame.surfarray.array3d(surface)              # (W, H, 3) uint8
        self._frames.append(arr.transpose(1, 0, 2).copy())  # (H, W, 3)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def encode(self) -> Optional[str]:
        """Encode buffered frames to a temp .mp4.  Returns the path, or None if empty."""
        if not self._frames:
            return None
        import imageio
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        writer = imageio.get_writer(
            tmp.name,
            fps=self.frame_rate,
            macro_block_size=1,   # no dimension constraint; yuv420p only needs even dims
            quality=self._quality,
        )
        try:
            for frame in self._frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return tmp.name
