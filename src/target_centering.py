# src/target_centering.py
"""
Target Centering and Autonomous Search Controller:
1. Target Centering Logic:
   - Computes normalized tracking errors (e_x, e_y) from frame center.
   - Steers pan/tilt angle commands smoothly using proportional damping.
2. Autonomous Search Pattern:
   - When target person is not in view / lost, actively sweeps camera pan angle
     to search the environment without physical hardware components.
3. Virtual PTZ HUD & Visualizer:
   - Draws tracking reticle, centering deadzone, error vectors, and simulated
     camera pan/tilt angle gauges directly onto the video frame.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .identity_state import TrackState


@dataclass
class CenteringState:
    pan_angle: float = 90.0
    tilt_angle: float = 90.0
    error_x: float = 0.0
    error_y: float = 0.0
    mode: str = "IDLE"  # "LOCK" | "HOLD_TEMPORARY" | "SEARCH" | "IDLE"
    is_centered: bool = False
    target_pos: Optional[Tuple[float, float]] = None


class TargetCenteringController:
    """
    Controls target centering and search motion for software-simulated
    or physical pan/tilt systems.
    """

    def __init__(
        self,
        min_pan: float = 20.0,
        max_pan: float = 160.0,
        center_pan: float = 90.0,
        min_tilt: float = 45.0,
        max_tilt: float = 135.0,
        center_tilt: float = 90.0,
        deadzone_frac: float = 0.08,
        pan_gain: float = 24.0,
        tilt_gain: float = 18.0,
        max_step_deg: float = 4.0,
        search_speed_deg_s: float = 35.0,
        lost_search_delay_s: float = 0.60,
    ):
        self.min_pan = min_pan
        self.max_pan = max_pan
        self.center_pan = center_pan
        self.min_tilt = min_tilt
        self.max_tilt = max_tilt
        self.center_tilt = center_tilt
        self.deadzone_frac = deadzone_frac
        self.pan_gain = pan_gain
        self.tilt_gain = tilt_gain
        self.max_step_deg = max_step_deg
        self.search_speed_deg_s = search_speed_deg_s
        self.lost_search_delay_s = lost_search_delay_s

        self.pan_angle = center_pan
        self.tilt_angle = center_tilt
        self.mode = "IDLE"
        self._last_target_time: float = 0.0
        self._last_update_time: float = time.time()
        self._search_dir: float = 1.0
        self._search_phase: float = 0.0

    def update(
        self,
        target_pos: Optional[Tuple[float, float]],
        frame_width: int,
        frame_height: int,
        track_state: TrackState = TrackState.UNKNOWN,
        allow_search: bool = True,
        timestamp: Optional[float] = None,
    ) -> CenteringState:
        now = time.time() if timestamp is None else float(timestamp)
        dt = max(0.001, min(0.2, now - self._last_update_time))
        self._last_update_time = now

        cx_frame = frame_width * 0.5
        cy_frame = frame_height * 0.5

        if target_pos is not None:
            tx, ty = target_pos
            err_x = (tx - cx_frame) / cx_frame
            err_y = (ty - cy_frame) / cy_frame
            self._last_target_time = now

            is_centered = (abs(err_x) <= self.deadzone_frac and abs(err_y) <= self.deadzone_frac)

            if track_state == TrackState.CONFIRMED:
                self.mode = "LOCK"
            elif track_state == TrackState.TEMPORARY:
                self.mode = "HOLD_TEMPORARY"
            else:
                self.mode = "TRACK_UNKNOWN"

            # Nudge pan and tilt toward target center
            if abs(err_x) > self.deadzone_frac:
                delta_pan = max(-self.max_step_deg, min(self.max_step_deg, self.pan_gain * err_x * dt * 15.0))
                self.pan_angle = max(self.min_pan, min(self.max_pan, self.pan_angle + delta_pan))

            if abs(err_y) > self.deadzone_frac:
                delta_tilt = max(-self.max_step_deg, min(self.max_step_deg, self.tilt_gain * err_y * dt * 15.0))
                self.tilt_angle = max(self.min_tilt, min(self.max_tilt, self.tilt_angle + delta_tilt))

            return CenteringState(
                pan_angle=self.pan_angle,
                tilt_angle=self.tilt_angle,
                error_x=err_x,
                error_y=err_y,
                mode=self.mode,
                is_centered=is_centered,
                target_pos=(tx, ty),
            )
        else:
            # Target is lost / absent
            time_lost = now - self._last_target_time
            if allow_search and time_lost >= self.lost_search_delay_s:
                self.mode = "SEARCH"
                # Autonomous sinusoidal / smooth sweep search pattern
                step = self.search_speed_deg_s * dt * self._search_dir
                self.pan_angle += step
                if self.pan_angle >= self.max_pan:
                    self.pan_angle = self.max_pan
                    self._search_dir = -1.0
                elif self.pan_angle <= self.min_pan:
                    self.pan_angle = self.min_pan
                    self._search_dir = 1.0

                # Gently return tilt to center
                self.tilt_angle += (self.center_tilt - self.tilt_angle) * 0.05
            else:
                self.mode = "IDLE" if time_lost > 3.0 else "HOLD_LOST"

            return CenteringState(
                pan_angle=self.pan_angle,
                tilt_angle=self.tilt_angle,
                error_x=0.0,
                error_y=0.0,
                mode=self.mode,
                is_centered=False,
                target_pos=None,
            )

    def draw_hud(
        self,
        vis: np.ndarray,
        state: CenteringState,
        target_name: Optional[str] = None,
        confidence_str: str = "",
    ) -> None:
        """
        Renders HUD with target centering crosshairs, deadzone boundaries,
        virtual pan/tilt gauges, and search motion telemetry.
        """
        H, W = vis.shape[:2]
        cx, cy = W // 2, H // 2

        # 1. Deadzone boundary box
        dz_w = int(self.deadzone_frac * W)
        dz_h = int(self.deadzone_frac * H)
        cv2.rectangle(
            vis,
            (cx - dz_w, cy - dz_h),
            (cx + dz_w, cy + dz_h),
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

        # 2. Frame Center Crosshair
        cv2.line(vis, (cx - 15, cy), (cx + 15, cy), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.line(vis, (cx, cy - 15), (cx, cy + 15), (120, 120, 120), 1, cv2.LINE_AA)

        # 3. Target lock vector & reticle
        if state.target_pos is not None:
            tx, ty = int(state.target_pos[0]), int(state.target_pos[1])
            if state.mode == "LOCK":
                color = (0, 255, 0)       # Confirmed Lock
            elif state.mode == "HOLD_TEMPORARY":
                color = (0, 190, 255)     # Temporary
            else:
                color = (0, 140, 255)

            # Error line from center to target
            cv2.line(vis, (cx, cy), (tx, ty), color, 1, cv2.LINE_AA)

            # Crosshair on target
            reticle_r = 16
            cv2.circle(vis, (tx, ty), reticle_r, color, 2, cv2.LINE_AA)
            cv2.line(vis, (tx - 24, ty), (tx - 8, ty), color, 2)
            cv2.line(vis, (tx + 8, ty), (tx + 24, ty), color, 2)
            cv2.line(vis, (tx, ty - 24), (tx, ty - 8), color, 2)
            cv2.line(vis, (tx, ty + 8), (tx, ty + 24), color, 2)

            if state.is_centered:
                cv2.putText(
                    vis, "TARGET CENTERED", (cx - 80, cy + dz_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
                )

        # 4. Status Banner
        mode_colors = {
            "LOCK": (0, 255, 0),
            "HOLD_TEMPORARY": (0, 190, 255),
            "SEARCH": (0, 140, 255),
            "HOLD_LOST": (0, 255, 255),
            "IDLE": (180, 180, 180),
            "TRACK_UNKNOWN": (200, 200, 200),
        }
        b_color = mode_colors.get(state.mode, (200, 200, 200))
        status_text = f"CAM MODE: {state.mode}"
        if state.mode == "SEARCH":
            status_text += f" (SWEEPING: {int(state.pan_angle)} deg)"
        elif target_name:
            status_text += f" -> {target_name} {confidence_str}"

        # Draw HUD bar at top right
        cv2.putText(
            vis, status_text, (W - 460, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            vis, status_text, (W - 460, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, b_color, 2, cv2.LINE_AA
        )

        # 5. Virtual Pan Angle Bar gauge at bottom
        bar_w = 240
        bar_x = (W - bar_w) // 2
        bar_y = H - 24
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (50, 50, 50), -1)
        pan_frac = (state.pan_angle - self.min_pan) / max(1.0, self.max_pan - self.min_pan)
        pan_frac = max(0.0, min(1.0, pan_frac))
        cursor_x = int(bar_x + pan_frac * bar_w)
        cv2.rectangle(vis, (cursor_x - 4, bar_y - 2), (cursor_x + 4, bar_y + 14), b_color, -1)
        cv2.putText(
            vis, f"PAN: {int(state.pan_angle)} deg", (bar_x - 110, bar_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
        )
