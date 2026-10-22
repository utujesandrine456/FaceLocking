# src/track_and_recognize.py
"""
Integrated Whole-Person Tracking + Face-Association + Three-State Identity Module.

Architecture:
  Webcam -> YOLOX-Nano (body dets) -> WholePersonTracker (IoU multi-object tracking)
         -> YuNet (face dets with 5pt landmarks) -> ArcFace ONNX (embeddings)
         -> FaceDBMatcher (cosine similarity) -> IdentityConfidence (3-state model)
         -> TargetCenteringController (soft pan/tilt + autonomous search)
         -> TachoPublisher (MQTT angle -> ESP8266 needle)

Three-State Model:
  CONFIRMED   : Face actively verified (cosine dist <= threshold, green).
  TEMPORARY   : Body tracked, face temporarily absent (head turn / occlusion).
                Confidence decays exponentially: c(t) = c0 * exp(-λ * dt).
                System says "TEMPORARY: <name>? (N%)" — honest uncertainty.
  UNKNOWN     : Confidence exhausted or no enrolled match.

Keys:
  q   : quit
  r   : reload face DB from disk
  +/- : adjust recognition threshold
  d   : toggle debug overlay
  s   : toggle autonomous search-when-lost (camera sweep)
  f   : toggle mirror flip
  m   : toggle MQTT publish

Run:
  python -m src.track_and_recognize
  python -m src.track_and_recognize --cam 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .yunet_5pt import align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .yolox_detector import YOLOXPersonDetector
from .body_tracker import WholePersonTracker, TrackedPerson, face_inside_body
from .identity_state import IdentityConfidence, TrackState
from .target_centering import TargetCenteringController, CenteringState
from .mqtt_servo import PanController
from .expression import analyze_optional, face_horizontal_side
from .recognize import (
    YuNetFaceDetector,
    FaceDBMatcher,
    FaceDet,
    load_db_npz,
    TachoPublisher,
)


def _draw_label_pill(img, text, pos, font_scale, color, bg_color=(30, 30, 30), pad_x=10, pad_y=6):
    """Draw text inside a pill-shaped semi-transparent background."""
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    px1 = x - pad_x
    py1 = y - th - pad_y
    px2 = x + tw + pad_x
    py2 = y + pad_y
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), bg_color, -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    # Colored left accent bar
    cv2.rectangle(img, (px1, py1 + 2), (px1 + 4, py2 - 2), color, -1)
    # Text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_track_overlay(
    vis: np.ndarray,
    track: TrackedPerson,
    is_primary: bool = False,
    show_debug: bool = False,
) -> None:
    """Renders a clean, modern overlay for one tracked person."""
    H, W = vis.shape[:2]
    x1, y1, x2, y2 = track.bbox
    state = track.identity.state
    conf = track.identity.confidence
    name = track.identity.name

    # --- Color scheme ---
    if state == TrackState.CONFIRMED:
        color = (0, 220, 80)         # Vibrant green
        frame_thick = 3
    elif state == TrackState.TEMPORARY:
        color = (0, 200, 255)        # Amber/yellow
        frame_thick = 2
    else:
        color = (80, 80, 200)        # Muted red
        frame_thick = 1

    # --- Corner bracket marks (modern tracking UI) ---
    corner_len = max(14, min(35, (x2 - x1) // 4))
    ct = frame_thick + 1
    # Top-left
    cv2.line(vis, (x1, y1), (x1 + corner_len, y1), color, ct, cv2.LINE_AA)
    cv2.line(vis, (x1, y1), (x1, y1 + corner_len), color, ct, cv2.LINE_AA)
    # Top-right
    cv2.line(vis, (x2, y1), (x2 - corner_len, y1), color, ct, cv2.LINE_AA)
    cv2.line(vis, (x2, y1), (x2, y1 + corner_len), color, ct, cv2.LINE_AA)
    # Bottom-left
    cv2.line(vis, (x1, y2), (x1 + corner_len, y2), color, ct, cv2.LINE_AA)
    cv2.line(vis, (x1, y2), (x1, y2 - corner_len), color, ct, cv2.LINE_AA)
    # Bottom-right
    cv2.line(vis, (x2, y2), (x2 - corner_len, y2), color, ct, cv2.LINE_AA)
    cv2.line(vis, (x2, y2), (x2, y2 - corner_len), color, ct, cv2.LINE_AA)

    # Thin full border (subtle)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Primary target: outer glow
    if is_primary and state != TrackState.UNKNOWN:
        cv2.rectangle(vis, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 1, cv2.LINE_AA)

    # --- Identity label pill above box ---
    # Round up to 100% if very close (>= 98%)
    disp_conf = 1.0 if conf >= 0.98 else conf
    if state == TrackState.CONFIRMED and name:
        label_text = f"  [LOCKED] {name}   {disp_conf:.0%}"
        bg = (15, 70, 25)
    elif state == TrackState.TEMPORARY and name:
        dt_s = track.identity.dt_since_face
        if disp_conf >= 0.70:   # Still high confidence — just turned away
            label_text = f"  {name}   {disp_conf:.0%}  (looking away)"
        else:                   # Confidence decaying — face lost a while ago
            label_text = f"  {name}?  {disp_conf:.0%}  lost {dt_s:.1f}s"
        bg = (15, 55, 75)
    else:
        label_text = "  UNKNOWN"
        bg = (55, 25, 25)

    label_y = max(y1 - 12, 22)
    _draw_label_pill(vis, label_text, (x1 + 2, label_y), 0.56, color, bg_color=bg)

    # --- Confidence bar (below label, above box top) ---
    bar_h = 4
    bar_w = x2 - x1
    bar_y = y1 + 3
    if bar_w > 10:
        fill_w = int(bar_w * max(0.0, min(1.0, conf)))
        cv2.rectangle(vis, (x1, bar_y), (x1 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        if fill_w > 0:
            cv2.rectangle(vis, (x1, bar_y), (x1 + fill_w, bar_y + bar_h), color, -1)

    # --- Face box ---
    if track.face_bbox is not None:
        fx1, fy1, fx2, fy2 = track.face_bbox
        cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), color, 1, cv2.LINE_AA)

    # --- Face keypoints ---
    if track.face_kps is not None:
        for (px, py) in track.face_kps.astype(int):
            cv2.circle(vis, (int(px), int(py)), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # --- Expression label ---
    if track.expression is not None:
        expr_txt = str(track.expression.label)
        cv2.putText(vis, expr_txt, (x1 + 4, min(H - 5, y2 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, expr_txt, (x1 + 4, min(H - 5, y2 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 230, 255), 1, cv2.LINE_AA)

    # --- TEMPORARY status (bottom of box) ---
    if state == TrackState.TEMPORARY:
        if disp_conf >= 0.70:
            warn = "LOOKING AWAY"
            warn_color = (0, 200, 255)    # Amber
        else:
            warn = "FACE TEMPORARILY LOST"
            warn_color = (0, 140, 200)    # Darker amber
        cv2.putText(vis, warn, (x1 + 4, min(H - 5, y2 + 32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, warn, (x1 + 4, min(H - 5, y2 + 32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, warn_color, 1, cv2.LINE_AA)

    # --- Debug: centroid trail ---
    if show_debug and len(track.centroid_history) > 1:
        pts = np.array(track.centroid_history, dtype=np.int32)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c_color = (int(alpha * color[0]), int(alpha * color[1]), int(alpha * color[2]))
            cv2.line(vis, tuple(pts[i - 1]), tuple(pts[i]), c_color, 1, cv2.LINE_AA)


def select_primary_track(
    tracks: List[TrackedPerson],
    prev_primary_id: Optional[int],
) -> Optional[TrackedPerson]:
    """Selects the highest-priority track to follow as the camera's main target."""
    if not tracks:
        return None

    # Priority: Confirmed > Temporary > Unknown; within same priority, prefer prev primary, then largest body area
    def rank(t: TrackedPerson):
        state_priority = {TrackState.CONFIRMED: 2, TrackState.TEMPORARY: 1, TrackState.UNKNOWN: 0}
        is_prev = 1 if t.track_id == prev_primary_id else 0
        return (state_priority.get(t.identity.state, 0), is_prev, t.bbox[2] * t.bbox[3])

    return max(tracks, key=rank)


def main():
    parser = argparse.ArgumentParser(
        description="Whole-person tracking + face-association + Three-State Identity"
    )
    parser.add_argument("--cam", type=int, default=1, help="Camera index (0=laptop, 1=USB)")
    parser.add_argument("--no-flip", action="store_true", help="Disable mirror flip")
    parser.add_argument("--search", action="store_true", default=False, help="Enable search on startup")
    args = parser.parse_args()

    db_path = Path("data/db/face_db.npz")

    print("[init] Loading YOLOX-Nano person detector...")
    body_det = YOLOXPersonDetector(
        model_path="models/yolox_nano.onnx",
        input_size=(416, 416),
        conf_thresh=0.25,
        nms_thresh=0.45,
    )

    print("[init] Loading YuNet face detector...")
    face_det = YuNetFaceDetector(smooth_alpha=0.55, debug=False)

    print("[init] Loading ArcFace ONNX embedder...")
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx", input_size=(112, 112), debug=False
    )

    print("[init] Loading face database...")
    db = load_db_npz(db_path)
    matcher = FaceDBMatcher(db=db, dist_thresh=0.55)
    print(f"[init] Loaded {len(matcher._names)} identities: {matcher._names}")

    print("[init] Initializing WholePersonTracker...")
    tracker = WholePersonTracker(
        max_lost_frames=60,       # ~10s at 6fps — keeps track alive through a full turn-around
        iou_match_thresh=0.15,    # Lower threshold = more robust body re-association
        decay_lambda=0.14,
        min_temporary_conf=0.15,
    )

    print("[init] Initializing TargetCenteringController...")
    centering = TargetCenteringController(
        deadzone_frac=0.07,
        pan_gain=26.0,
        tilt_gain=20.0,
        max_step_deg=4.0,
        search_speed_deg_s=35.0,
        lost_search_delay_s=0.55,
    )

    # Try MQTT tacho publisher (non-blocking)
    tacho_pub: Optional[TachoPublisher] = None
    try:
        tacho_pub = TachoPublisher(broker="10.12.73.141", port=1883)
        tacho_pub.connect()
        print("[init] MQTT tacho publisher connected.")
    except Exception as e:
        print(f"[init] MQTT disabled: {e}")

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}. Try --cam 0")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mirror = not args.no_flip
    search_enabled = args.search
    show_debug = False
    mqtt_enabled = False
    primary_track_id: Optional[int] = None

    print("\n=== TRACKING STARTED ===")
    print("q=quit | r=reload DB | +/- threshold | d=debug | s=search | f=flip | m=mqtt")
    print(f"Mirror: {'ON' if mirror else 'OFF'} | Search-on-lost: {'ON' if search_enabled else 'OFF'}")

    t0 = time.time()
    frames = 0
    fps = 0.0
    centering_state = CenteringState()

    # --- Performance: skip body detection on alternate frames ---
    frame_count = 0
    BODY_DETECT_INTERVAL = 2   # run YOLOX every N frames (1 = every frame)
    prev_body_dets = []

    while True:
        # --- Flush camera buffer: grab latest frame, discard stale ones ---
        for _ in range(2):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        now = time.time()
        H, W = frame.shape[:2]
        vis = frame.copy()
        frame_count += 1

        # ---- 1. Whole-person body detection (YOLOX) — skip on some frames ----
        if frame_count % BODY_DETECT_INTERVAL == 0 or not prev_body_dets:
            body_dets = body_det.detect(frame)
            prev_body_dets = body_dets
        else:
            body_dets = prev_body_dets

        # ---- 2. Face detection + embedding + matching ----
        face_dets = face_det.detect(frame, max_faces=5)
        face_matches = []
        for fd in face_dets:
            try:
                aligned, _ = align_face_5pt(frame, fd.kps, out_size=(112, 112))
                emb = embedder.embed(aligned).embedding
                mr = matcher.match(emb)
                face_bbox = (fd.x1, fd.y1, fd.x2, fd.y2)
                face_matches.append((
                    face_bbox,
                    fd.kps,
                    mr.name if mr.accepted else None,
                    mr.similarity,
                    mr.distance,
                    fd.expression,
                ))
            except Exception:
                continue

        # ---- 3. Update multi-person body tracker + face-body association ----
        active_tracks = tracker.update(
            body_dets=body_dets,
            face_matches=face_matches,
            timestamp=now,
        )

        # ---- 4. Select primary (camera target) track ----
        primary = select_primary_track(active_tracks, primary_track_id)
        if primary is not None:
            primary_track_id = primary.track_id

        # ---- 5. Target centering + autonomous search ----
        target_pos: Optional[Tuple[float, float]] = None
        if primary is not None:
            cx_body, cy_body = primary.center
            target_pos = (cx_body, cy_body)

        centering_state = centering.update(
            target_pos=target_pos,
            frame_width=W,
            frame_height=H,
            track_state=primary.identity.state if primary else TrackState.UNKNOWN,
            allow_search=search_enabled,
            timestamp=now,
        )

        # ---- 6. MQTT needle update ----
        if tacho_pub is not None:
            is_confirmed = (
                primary is not None
                and primary.identity.state == TrackState.CONFIRMED
            )
            tacho_pub.update(
                face_cx=target_pos[0] if target_pos else None,
                face_cy=target_pos[1] if target_pos else None,
                frame_width=W,
                frame_height=H,
                is_locked=is_confirmed,
                axis="vertical",
            )

        # ---- 7. Draw all tracked persons ----
        for track in active_tracks:
            is_primary = (primary is not None and track.track_id == primary.track_id)
            draw_track_overlay(vis, track, show_debug=show_debug)
            if is_primary:
                # Outer highlight ring on primary target
                x1, y1, x2, y2 = track.bbox
                cv2.rectangle(vis, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                               track.identity.display_color, 1, cv2.LINE_AA)

        # ---- 8. Draw HUD (centering crosshair, pan gauge, mode status) ----
        conf_str = ""
        target_name = None
        if primary is not None:
            target_name = primary.identity.name
            hud_conf = primary.identity.confidence
            hud_conf = 1.0 if hud_conf >= 0.98 else hud_conf
            conf_str = f"({hud_conf:.0%})"
        centering.draw_hud(vis, centering_state, target_name=target_name, confidence_str=conf_str)

        # ---- 9. Info overlay header ----
        frames += 1
        dt = now - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = now

        n_confirmed = sum(1 for t in active_tracks if t.identity.state == TrackState.CONFIRMED)
        n_temporary = sum(1 for t in active_tracks if t.identity.state == TrackState.TEMPORARY)
        n_unknown = sum(1 for t in active_tracks if t.identity.state == TrackState.UNKNOWN)

        header = (
            f"cam={args.cam}  fps={fps:.1f}  bodies={len(active_tracks)}"
            f"  [C={n_confirmed} T={n_temporary} U={n_unknown}]"
            f"  thr={matcher.dist_thresh:.2f}"
            f"  IDs={len(matcher._names)}"
        )
        flags = []
        if mirror: flags.append("FLIP")
        if search_enabled: flags.append("SEARCH")
        if mqtt_enabled: flags.append("MQTT")
        if flags: header += "  " + "|".join(flags)

        cv2.putText(vis, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 90), 1, cv2.LINE_AA)

        cv2.imshow("Whole-Person Track & Recognize [3-State]", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            matcher.reload_from(db_path)
            print(f"[reload] DB reloaded: {len(matcher._names)} identities: {matcher._names}")
        elif key in (ord("+"), ord("=")):
            matcher.dist_thresh = min(1.20, matcher.dist_thresh + 0.01)
            print(f"[threshold] dist={matcher.dist_thresh:.2f}")
        elif key == ord("-"):
            matcher.dist_thresh = max(0.05, matcher.dist_thresh - 0.01)
            print(f"[threshold] dist={matcher.dist_thresh:.2f}")
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"[debug] {'ON' if show_debug else 'OFF'}")
        elif key == ord("s"):
            search_enabled = not search_enabled
            print(f"[search] {'ON - camera will sweep when target lost' if search_enabled else 'OFF'}")
        elif key == ord("f"):
            mirror = not mirror
            print(f"[flip] mirror={'ON' if mirror else 'OFF'}")
        elif key == ord("m"):
            mqtt_enabled = not mqtt_enabled
            print(f"[mqtt] {'ON' if mqtt_enabled else 'OFF'}")

    if tacho_pub is not None:
        tacho_pub.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[done] Tracking session ended.")


if __name__ == "__main__":
    main()
