# src/identity_state.py
"""
Honest Three-State Model with Exponential Confidence Decay:
- CONFIRMED: Face actively detected & verified with high cosine similarity (d <= d_thresh).
- TEMPORARY: Face temporarily lost (head turn, occlusion, looking away) while body tracking
             is maintained. Confidence decays exponentially: c(t) = c0 * exp(-lambda * dt).
             Identity continuity is preserved during transient loss.
- UNKNOWN:   Confidence decayed below threshold (c(t) < c_min), or person has no matched identity.
"""
from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


class TrackState(str, enum.Enum):
    CONFIRMED = "CONFIRMED"
    TEMPORARY = "TEMPORARY"
    UNKNOWN = "UNKNOWN"


@dataclass
class IdentityConfidence:
    """
    Manages the honest three-state confidence lifecycle for a tracked target.
    """
    name: Optional[str] = None
    state: TrackState = TrackState.UNKNOWN
    confidence: float = 0.0          # Current effective confidence [0.0, 1.0]
    base_confidence: float = 0.0     # c0 (confidence at last confirmation)
    last_confirmed_time: float = 0.0 # Timestamp of last positive face recognition
    last_seen_time: float = 0.0      # Timestamp of last body/face detection
    decay_lambda: float = 0.14       # Exponential decay (half-life ~5s — survives a full turn-around)
    min_temporary_conf: float = 0.15 # Below this, drops from TEMPORARY to UNKNOWN (~20s at 90% base)
    confirmed_thresh: float = 0.42   # Cosine similarity threshold for CONFIRMED
    confirmed_grace_s: float = 3.0   # Stay CONFIRMED for 3s after last face match (covers a full turn)
    best_similarity: float = 0.0
    best_distance: float = 1.0
    frames_since_face: int = 0
    total_face_matches: int = 0

    @staticmethod
    def _boost_similarity(sim: float, thresh: float) -> float:
        """Map raw cosine similarity to a user-friendly confidence percentage.

        The raw cosine similarity from ArcFace is typically 0.4-0.8 even for
        a perfect match. This maps it to 90-100% when above the accept
        threshold so the display is intuitive: person present → ~100%.

        Below threshold the raw value passes through as-is.
        """
        if sim >= thresh:
            # Linear map: [thresh, 1.0] → [0.90, 1.0]
            ratio = min(1.0, (sim - thresh) / max(0.01, 1.0 - thresh))
            return 0.90 + 0.10 * ratio
        return float(sim)

    def confirm_face(
        self,
        name: str,
        similarity: float,
        distance: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Called when a high-confidence face match is detected on this track."""
        now = time.time() if timestamp is None else float(timestamp)
        self.name = name
        self.best_similarity = float(similarity)
        self.best_distance = float(distance)
        # Boost similarity into a user-friendly confidence (>= 90% when accepted)
        self.base_confidence = self._boost_similarity(float(similarity), self.confirmed_thresh)
        self.confidence = self.base_confidence
        self.last_confirmed_time = now
        self.last_seen_time = now
        self.frames_since_face = 0
        self.total_face_matches += 1
        self.state = TrackState.CONFIRMED

    def update_without_face(self, timestamp: Optional[float] = None) -> TrackState:
        """
        Called on every frame where the body is tracked but face is absent
        (e.g., head turn, occlusion, back turned). Applies exponential confidence decay:
            c(t) = c0 * exp(-lambda * dt)

        Grace period: stays CONFIRMED for `confirmed_grace_s` seconds after
        the last face match, so a single missed frame doesn't flicker the state.
        """
        now = time.time() if timestamp is None else float(timestamp)
        self.last_seen_time = now
        self.frames_since_face += 1

        if self.state in (TrackState.CONFIRMED, TrackState.TEMPORARY) and self.name is not None:
            dt = max(0.0, now - self.last_confirmed_time)

            # Grace period: keep CONFIRMED for a short window even without face
            if dt <= self.confirmed_grace_s and self.state == TrackState.CONFIRMED:
                # Confidence holds steady during grace period
                self.confidence = self.base_confidence
                return self.state

            # After grace period, begin exponential confidence decay
            dt_decay = max(0.0, dt - self.confirmed_grace_s)
            decay_factor = math.exp(-self.decay_lambda * dt_decay)
            self.confidence = float(self.base_confidence * decay_factor)

            if self.confidence >= self.min_temporary_conf:
                self.state = TrackState.TEMPORARY
            else:
                self.state = TrackState.UNKNOWN
                self.confidence = 0.0
        else:
            self.state = TrackState.UNKNOWN
            self.confidence = 0.0

        return self.state

    def mark_lost(self) -> None:
        """Called when the entire body is lost."""
        self.state = TrackState.UNKNOWN
        self.confidence = 0.0

    @property
    def dt_since_face(self) -> float:
        if self.last_confirmed_time <= 0:
            return 999.0
        return max(0.0, time.time() - self.last_confirmed_time)

    @property
    def display_label(self) -> str:
        if self.state == TrackState.CONFIRMED:
            return f"CONFIRMED: {self.name} ({self.confidence:.0%})"
        elif self.state == TrackState.TEMPORARY:
            return f"TEMPORARY: {self.name}? ({self.confidence:.0%}, dt={self.dt_since_face:.1f}s)"
        else:
            return "UNKNOWN"

    @property
    def display_color(self) -> Tuple[int, int, int]:
        """BGR color: Green for Confirmed, Amber/Yellow for Temporary, Red for Unknown."""
        if self.state == TrackState.CONFIRMED:
            return (0, 255, 0)      # Vibrant green
        elif self.state == TrackState.TEMPORARY:
            return (0, 190, 255)    # Amber / bright orange-yellow
        else:
            return (0, 0, 255)      # Red
