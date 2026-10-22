# src/body_tracker.py
"""
Whole-Person Multi-Object Tracker & Spatial Face-to-Body Association:
1. Tracks human body bounding boxes frame-to-frame using IoU & smooth centroid filtering.
2. Spatially associates detected faces to corresponding human bodies.
3. Maintains temporal identity continuity via the Honest Three-State Model
   across temporary facial loss (head turns, occlusions, path crossings).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .identity_state import IdentityConfidence, TrackState
from .yolox_detector import BodyDet


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0

    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return float(interArea) / float(boxAArea + boxBArea - interArea)


def face_inside_body(
    face_box: Tuple[int, int, int, int],
    body_box: Tuple[int, int, int, int],
    max_face_y_ratio: float = 0.65,
) -> float:
    """
    Computes spatial compatibility between a face and body bounding box.
    Returns a score [0.0, 1.0], where > 0 indicates spatial containment.
    """
    fx1, fy1, fx2, fy2 = face_box
    bx1, by1, bx2, by2 = body_box
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    fcx = 0.5 * (fx1 + fx2)
    fcy = 0.5 * (fy1 + fy2)

    # Face center must be inside the body horizontal span (with small margin)
    margin_x = 0.15 * bw
    if not (bx1 - margin_x <= fcx <= bx2 + margin_x):
        return 0.0

    # Face must be in the upper portion of the body (head/shoulders)
    max_y = by1 + max_face_y_ratio * bh
    min_y = by1 - 0.20 * bh
    if not (min_y <= fcy <= max_y):
        return 0.0

    # Calculate intersection area over face area
    ix1 = max(fx1, bx1)
    iy1 = max(fy1, by1)
    ix2 = min(fx2, bx2)
    iy2 = min(fy2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
    containment = float(inter) / float(face_area)

    return containment


@dataclass
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    identity: IdentityConfidence = field(default_factory=IdentityConfidence)
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_kps: Optional[np.ndarray] = None
    expression: Optional[object] = None
    time_since_update: int = 0
    total_frames_tracked: int = 1
    centroid_history: List[Tuple[int, int]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.bbox[0] + self.bbox[2]), 0.5 * (self.bbox[1] + self.bbox[3]))

    def update_position(self, new_bbox: Tuple[int, int, int, int], score: float) -> None:
        prev_cx, prev_cy = self.center
        cur_cx = 0.5 * (new_bbox[0] + new_bbox[2])
        cur_cy = 0.5 * (new_bbox[1] + new_bbox[3])
        self.velocity = (cur_cx - prev_cx, cur_cy - prev_cy)

        # Smooth bbox update with EMA
        alpha = 0.70
        smoothed = [
            int(round(alpha * self.bbox[i] + (1.0 - alpha) * new_bbox[i]))
            for i in range(4)
        ]
        self.bbox = (smoothed[0], smoothed[1], smoothed[2], smoothed[3])
        self.score = score
        self.time_since_update = 0
        self.total_frames_tracked += 1

        self.centroid_history.append((int(cur_cx), int(cur_cy)))
        if len(self.centroid_history) > 30:
            self.centroid_history.pop(0)


class WholePersonTracker:
    """
    Maintains temporal tracking of human bodies and spatially associated faces.
    """

    def __init__(
        self,
        max_lost_frames: int = 15,
        iou_match_thresh: float = 0.25,
        decay_lambda: float = 0.45,
        min_temporary_conf: float = 0.28,
    ):
        self.max_lost_frames = max_lost_frames
        self.iou_match_thresh = iou_match_thresh
        self.decay_lambda = decay_lambda
        self.min_temporary_conf = min_temporary_conf
        self.tracks: Dict[int, TrackedPerson] = {}
        self._next_id: int = 1

    def update(
        self,
        body_dets: List[BodyDet],
        face_matches: List[Tuple[Tuple[int, int, int, int], np.ndarray, Optional[str], float, float, Optional[object]]],
        timestamp: Optional[float] = None,
    ) -> List[TrackedPerson]:
        """
        Updates tracking state with new detections and face matches.
        face_matches item: (face_bbox, kps, name, similarity, distance, expression)
        """
        now = time.time() if timestamp is None else float(timestamp)

        # 1. Match detected bodies to existing tracks using IoU
        matched_track_ids = set()
        matched_det_indices = set()

        if self.tracks and body_dets:
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(body_dets)), dtype=np.float32)

            for i, tid in enumerate(track_ids):
                for j, bdet in enumerate(body_dets):
                    iou_matrix[i, j] = compute_iou(self.tracks[tid].bbox, bdet.box_xyxy)

            # Greedy IoU assignment
            while True:
                max_val = float(np.max(iou_matrix))
                if max_val < self.iou_match_thresh:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                if tid not in matched_track_ids and j not in matched_det_indices:
                    self.tracks[tid].update_position(body_dets[j].box_xyxy, body_dets[j].score)
                    matched_track_ids.add(tid)
                    matched_det_indices.add(j)
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # 2. Spawn new tracks for unmatched body detections
        for j, bdet in enumerate(body_dets):
            if j not in matched_det_indices:
                tid = self._next_id
                self._next_id += 1
                tperson = TrackedPerson(
                    track_id=tid,
                    bbox=bdet.box_xyxy,
                    score=bdet.score,
                    identity=IdentityConfidence(
                        decay_lambda=self.decay_lambda,
                        min_temporary_conf=self.min_temporary_conf,
                    ),
                    centroid_history=[(int(0.5 * (bdet.x1 + bdet.x2)), int(0.5 * (bdet.y1 + bdet.y2)))],
                )
                self.tracks[tid] = tperson
                matched_track_ids.add(tid)

        # 3. Associate detected faces to tracks spatially
        assigned_faces = set()
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            best_face_idx = None
            best_overlap = 0.0

            for f_idx, (f_box, kps, name, sim, dist, expr) in enumerate(face_matches):
                if f_idx in assigned_faces:
                    continue
                overlap = face_inside_body(f_box, track.bbox)
                if overlap > 0.30 and overlap > best_overlap:
                    best_overlap = overlap
                    best_face_idx = f_idx

            if best_face_idx is not None:
                assigned_faces.add(best_face_idx)
                f_box, kps, name, sim, dist, expr = face_matches[best_face_idx]
                track.face_bbox = f_box
                track.face_kps = kps
                track.expression = expr

                if name is not None and sim >= track.identity.confirmed_thresh:
                    track.identity.confirm_face(name, sim, dist, timestamp=now)
                else:
                    # Face is visible but unknown or ambiguous
                    track.identity.update_without_face(timestamp=now)
            else:
                # No face detected on this body (head turn, occlusion, or back turned)
                track.face_bbox = None
                track.face_kps = None
                track.expression = None
                track.identity.update_without_face(timestamp=now)

        # 3b. Create virtual body tracks for unassigned faces (face visible but
        #     no YOLOX body box contains it — e.g. person too close to camera).
        for f_idx, (f_box, kps, name, sim, dist, expr) in enumerate(face_matches):
            if f_idx in assigned_faces:
                continue
            # Expand face box into a synthetic body region
            fx1, fy1, fx2, fy2 = f_box
            fw = max(1, fx2 - fx1)
            fh = max(1, fy2 - fy1)
            # Body is roughly 3x taller than face and 1.8x wider, centered on face
            bw = int(fw * 1.8)
            bh = int(fh * 3.0)
            bcx = (fx1 + fx2) // 2
            bx1 = max(0, bcx - bw // 2)
            bx2 = bx1 + bw
            by1 = max(0, fy1 - int(fh * 0.3))
            by2 = by1 + bh
            synth_bbox = (bx1, by1, bx2, by2)

            # Try to match to an existing track by IoU with the synthetic box
            best_tid = None
            best_iou = 0.0
            for tid, track in self.tracks.items():
                iou = compute_iou(track.bbox, synth_bbox)
                if iou > 0.10 and iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                track = self.tracks[best_tid]
            else:
                # Spawn a new track from the synthetic body
                tid = self._next_id
                self._next_id += 1
                track = TrackedPerson(
                    track_id=tid,
                    bbox=synth_bbox,
                    score=0.5,
                    identity=IdentityConfidence(
                        decay_lambda=self.decay_lambda,
                        min_temporary_conf=self.min_temporary_conf,
                    ),
                    centroid_history=[(bcx, (by1 + by2) // 2)],
                )
                self.tracks[tid] = track
                matched_track_ids.add(tid)

            track.face_bbox = f_box
            track.face_kps = kps
            track.expression = expr
            if name is not None and sim >= track.identity.confirmed_thresh:
                track.identity.confirm_face(name, sim, dist, timestamp=now)
            else:
                track.identity.update_without_face(timestamp=now)

        # 4. Handle tracks that were not matched to any body detection in this frame
        dead_tracks = []
        for tid, track in self.tracks.items():
            if tid not in matched_track_ids:
                track.time_since_update += 1
                track.identity.update_without_face(timestamp=now)
                if track.time_since_update > self.max_lost_frames:
                    dead_tracks.append(tid)

        for tid in dead_tracks:
            del self.tracks[tid]

        return list(self.tracks.values())
