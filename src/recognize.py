# src/recognize.py
"""
Multi-face recognition (CPU-friendly):
YuNet (box + 5pt) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.

Run: python -m src.recognize
     python -m src.recognize --cam 1   # use USB / embedded camera (not laptop)

Keys:
  q : quit
  r : reload DB from disk (data/db/face_db.npz)
  +/- : adjust threshold (distance) live
  d : toggle debug overlay
  i : invert pan mapping (if servo turns the wrong way)
  m : toggle MQTT pan publish on/off
  s : toggle search-when-lost on/off
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .yunet_5pt import YuNet5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .mqtt_servo import ServoPanPublisher, PanController
from .expression import ExpressionState, FaceSideState, analyze_optional, face_horizontal_side

try:
    import paho.mqtt.client as _paho_mqtt
except Exception:
    _paho_mqtt = None


class TachoPublisher:
    """Publishes angle (0-180) to ESP8266 stepper when a known face is locked.

    Simple rule:
      - Known face LOCKED   -> publish angle matching face X position.
                               Needle follows in real time.
      - Face NOT seen / unknown -> publish NOTHING.
                                   Needle stays frozen at its last position.

    MQTT topic : face/status
    Payload    : ASCII integer  e.g. "0" (left) | "90" (centre) | "180" (right)
    """

    TOPIC_STATUS = "face/status"
    TOPIC_TACHO  = "tacho/state"
    ANGLE_MIN    = 0
    ANGLE_MAX    = 180

    def __init__(self, broker: str = "10.12.73.141", port: int = 1883):
        if _paho_mqtt is None:
            raise RuntimeError("paho-mqtt not installed. Run: pip install paho-mqtt")
        self.broker = broker
        self.port   = port
        self._connected  = False
        self._last_angle: Optional[int] = None

        if hasattr(_paho_mqtt, "CallbackAPIVersion"):
            self._client = _paho_mqtt.Client(
                callback_api_version=_paho_mqtt.CallbackAPIVersion.VERSION1,
                client_id="pc-face-tacho",
            )
        else:
            self._client = _paho_mqtt.Client(client_id="pc-face-tacho")

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = rc == 0
        if self._connected:
            print(f"[tacho] MQTT connected {self.broker}:{self.port}")
            self._client.publish(self.TOPIC_TACHO, "online", retain=True)
        else:
            print(f"[tacho] MQTT connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print(f"[tacho] MQTT disconnected rc={rc}")

    def connect(self) -> None:
        import time as _time
        self._client.connect(self.broker, self.port, keepalive=30)
        self._client.loop_start()
        deadline = _time.time() + 3.0
        while _time.time() < deadline and not self._connected:
            _time.sleep(0.05)
        if not self._connected:
            print("[tacho] warning: not connected yet; retrying in background")

    def _send_angle(self, angle: int) -> None:
        """Raw publish — only sends if angle actually changed."""
        angle = int(max(self.ANGLE_MIN, min(self.ANGLE_MAX, angle)))
        if angle == self._last_angle:
            return
        self._last_angle = angle
        info = self._client.publish(self.TOPIC_STATUS, str(angle), qos=1)
        ok   = (info.rc == 0)
        print(f"[tacho] face/status -> '{angle}' {'ok' if ok else 'FAIL'}")

    def update(
        self,
        face_cx: Optional[float],
        face_cy: Optional[float],
        frame_width: int,
        frame_height: int,
        is_locked: bool,
        axis: str = "vertical",
    ) -> None:
        """
        Call every frame.

        face_cx      : horizontal centre of tracked face in pixels (None = no face)
        face_cy      : vertical centre of tracked face in pixels (None = no face)
        frame_width  : full frame width in pixels
        frame_height : full frame height in pixels
        is_locked    : True ONLY when a known (enrolled) face is locked
        axis         : 'vertical' (head UP -> needle LEFT) or 'horizontal'
        """
        if not is_locked:
            # Face not seen or identity unknown -> do nothing.
            # Needle stays frozen at its last position.
            return

        if axis == "vertical":
            if face_cy is None:
                return
            # y=0 is top (head up -> ratio ~ 0 -> needle 0 deg left)
            # y=h is bottom (head down -> ratio ~ 1 -> needle 180 deg right)
            ratio = float(face_cy) / float(max(frame_height, 1))
        else:
            if face_cx is None:
                return
            ratio = float(face_cx) / float(max(frame_width, 1))

        ratio = max(0.0, min(1.0, ratio))
        angle = int(round(self.ANGLE_MIN + ratio * (self.ANGLE_MAX - self.ANGLE_MIN)))
        self._send_angle(angle)

    def close(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords
    expression: Optional[ExpressionState] = None


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
    return out


class YuNetFaceDetector(YuNet5ptDetector):
    """YuNet detector that returns FaceDet (optional expression field)."""

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        boxes = super().detect(frame_bgr, max_faces=max_faces)
        return [
            FaceDet(
                x1=f.x1,
                y1=f.y1,
                x2=f.x2,
                y2=f.y2,
                score=f.score,
                kps=f.kps,
                expression=None,
            )
            for f in boxes
        ]


# Backward-compatible alias
HaarFaceMesh5pt = YuNetFaceDetector


class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.45):
        self.db = db
        self.dist_thresh = float(dist_thresh)
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self):
        self._names = sorted(self.db.keys())
        if self._names:
            self._mat = np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
        else:
            self._mat = None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None or len(self._names) == 0:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)

        e = emb.reshape(1, -1).astype(np.float32)
        sims = (self._mat @ e.T).reshape(-1)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_dist = 1.0 - best_sim
        ok = best_dist <= self.dist_thresh

        return MatchResult(
            name=self._names[best_i] if ok else None,
            distance=float(best_dist),
            similarity=float(best_sim),
            accepted=bool(ok),
        )


def main():
    parser = argparse.ArgumentParser(description="Live face recognition + MQTT servo pan")
    parser.add_argument(
        "--cam",
        type=int,
        default=1,
        help="OpenCV camera index (default 1 = usually USB/embedded; 0 = laptop webcam)",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable horizontal mirror flip (mirror is enabled by default)",
    )
    parser.add_argument(
        "--axis",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Tracking axis: 'vertical' (head UP -> needle LEFT) or 'horizontal' (default: vertical)",
    )
    args = parser.parse_args()

    db_path = Path("data/db/face_db.npz")

    det = YuNetFaceDetector(smooth_alpha=0.65, debug=False)
    embedder = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112), debug=False)

    db = load_db_npz(db_path)
    # Slightly looser threshold helps smile / blink variation after diverse enrollment.
    matcher = FaceDBMatcher(db=db, dist_thresh=0.45)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Camera index {args.cam} not available. "
            "Find the right one with: .\\.venv\\Scripts\\python.exe -m src.camera --list"
        )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Servo pan publisher is DISABLED — hardware is a stepper motor, not a servo.
    # All needle control goes through TachoPublisher (face/status topic).
    mqtt_enabled   = False   # keeps ServoPanPublisher silent
    search_enabled = False   # needle never sweeps — it only follows a locked face
    invert_pan     = False
    pan_angle: Optional[int] = 90
    pan_mode = "idle"
    pan = PanController(
        deadzone_frac=0.07,
        track_gain=22.0,
        max_step_deg=3.0,
        lost_before_search_s=0.45,
        search_step_deg=5.0,
        search_period_s=0.10,
    )
    servo_pub: Optional[ServoPanPublisher] = None
    locked_label: Optional[str] = None
    # Hold last accepted identity briefly through blinks / expression changes.
    sticky_name: Optional[str] = None
    sticky_until = 0.0
    sticky_hold_s = 0.7
    last_expr_log: Optional[str] = None
    last_side_log: Optional[str] = None

    # --- Tachometer publisher (sends "1"/"0" to face/status for ESP8266 stepper) ---
    tacho_pub: Optional[TachoPublisher] = None
    try:
        tacho_pub = TachoPublisher(broker="10.12.73.141", port=1883)
        tacho_pub.connect()
        print("[tacho] publisher ready — needle will track once your face is locked")
    except Exception as e:
        tacho_pub = None
        print(f"[tacho] MQTT disabled: {e}")

    # ServoPanPublisher intentionally not started (no servo in this setup).
    servo_pub: Optional[ServoPanPublisher] = None

    print(
        f"Recognize cam={args.cam}. "
        "q=quit  r=reload  +/- threshold  d=debug  i=invert  f=flip-mirror  v=axis(V/H)"
    )
    print("[tacho] Needle at CENTER on ESP startup. Move into view to be identified, then move and needle follows.")

    mirror = not args.no_flip
    print(f"[recognize] mirror flip (horizontal): {'ON' if mirror else 'OFF'}")

    track_axis = args.axis
    print(f"[tacho] tracking axis: {track_axis.upper()} (head UP -> needle LEFT, head DOWN -> needle RIGHT)")

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    show_debug = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        faces = det.detect(frame, max_faces=5)
        vis = frame.copy()

        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        h, w = vis.shape[:2]
        thumb = 112
        pad = 8
        x0 = w - thumb - pad
        y0 = 80
        shown = 0

        # Prefer a known face for pan locking; else use the first detection.
        track_face: Optional[FaceDet] = None
        track_label = "none"
        first_face: Optional[FaceDet] = None
        first_label = "none"
        known_face: Optional[FaceDet] = None
        known_label = "none"

        for i, f in enumerate(faces):
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = embedder.embed(aligned).embedding
            mr = matcher.match(emb)

            now_t = time.time()
            if mr.accepted and mr.name is not None:
                sticky_name = mr.name
                sticky_until = now_t + sticky_hold_s
                label = mr.name
                accepted = True
            elif sticky_name is not None and now_t < sticky_until:
                # Keep identity through blink / short expression change.
                label = sticky_name
                accepted = True
            else:
                label = mr.name if mr.name is not None else "Unknown"
                accepted = bool(mr.accepted)

            line1 = f"{label}"
            line2 = f"dist={mr.distance:.3f} sim={mr.similarity:.3f}"

            color = (0, 255, 0) if accepted else (0, 0, 255)
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 2, color, -1)
            cv2.putText(vis, line1, (f.x1, max(0, f.y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(vis, line2, (f.x1, max(0, f.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if f.expression is not None:
                face_mid_x = 0.5 * (f.x1 + f.x2)
                side = face_horizontal_side(face_mid_x, w, center_frac=0.12)
                expr_txt = f"{f.expression.label} | {side.label}"
                cv2.putText(
                    vis, expr_txt, (f.x1, min(h - 10, f.y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                )
                log_key = f"{label}|{expr_txt}"
                if log_key != last_expr_log:
                    last_expr_log = log_key
                    print(
                        f"[expression] {label}: {f.expression.as_log()}, {side.as_log()} "
                        f"(smile={f.expression.smile_score:.2f}, ear={f.expression.eye_openness:.2f}, "
                        f"offset={side.offset:+.2f})"
                    )
                if f.expression.blinked:
                    print(f"[blink] {label}: BLINK detected (count={f.expression.blink_count})")
                    cv2.putText(
                        vis, "BLINK!", (f.x1, min(h - 10, f.y2 + 44)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2,
                    )
                if side.side != last_side_log:
                    last_side_log = side.side
                    print(f"[position] {label}: {side.as_log()}")
            else:
                face_mid_x = 0.5 * (f.x1 + f.x2)
                side = face_horizontal_side(face_mid_x, w, center_frac=0.12)
                cv2.putText(
                    vis, f"side: {side.label}", (f.x1, min(h - 10, f.y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                )
                if side.side != last_side_log:
                    last_side_log = side.side
                    print(f"[position] {label}: {side.as_log()}")

            if y0 + thumb <= h and shown < 4:
                vis[y0:y0 + thumb, x0:x0 + thumb] = aligned
                cv2.putText(vis, f"{i+1}:{label}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y0 += thumb + pad
                shown += 1

            if show_debug:
                dbg = f"kpsLeye=({f.kps[0,0]:.0f},{f.kps[0,1]:.0f})"
                cv2.putText(vis, dbg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if first_face is None:
                first_face = f
                first_label = label
            if accepted and known_face is None:
                known_face = f
                known_label = label
            # Stick to previously locked identity when possible.
            if locked_label is not None and label == locked_label:
                track_face = f
                track_label = label

        if track_face is None:
            if known_face is not None:
                track_face = known_face
                track_label = known_label
            elif first_face is not None:
                track_face = first_face
                track_label = first_label

        face_cx: Optional[float] = None
        face_cy: Optional[float] = None
        if track_face is not None:
            face_cx = 0.5 * (track_face.x1 + track_face.x2)
            face_cy = 0.5 * (track_face.y1 + track_face.y2)

            if track_axis == "vertical":
                # Horizontal guideline tracking vertical face movement
                cv2.line(vis, (0, int(face_cy)), (w, int(face_cy)), (255, 200, 0), 2)
                cv2.line(vis, (0, h // 2), (w, h // 2), (0, 255, 255), 1)
                dz = int(0.04 * h)
                cv2.line(vis, (0, h // 2 - dz), (w, h // 2 - dz), (80, 80, 80), 1)
                cv2.line(vis, (0, h // 2 + dz), (w, h // 2 + dz), (80, 80, 80), 1)
            else:
                # Vertical guideline tracking horizontal face movement
                cv2.line(vis, (int(face_cx), 0), (int(face_cx), h), (255, 200, 0), 2)
                cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
                dz = int(0.04 * w)
                cv2.line(vis, (w // 2 - dz, 0), (w // 2 - dz, h), (80, 80, 80), 1)
                cv2.line(vis, (w // 2 + dz, 0), (w // 2 + dz, h), (80, 80, 80), 1)

        pan.set_invert(invert_pan)
        prev_angle = int(pan_angle) if pan_angle is not None else None
        pan_angle, pan_mode = pan.update(face_cx, w, allow_search=search_enabled)

        if pan_mode == "lock" and track_label != "none":
            locked_label = track_label
        elif pan_mode in ("search", "idle"):
            locked_label = None

        if mqtt_enabled and servo_pub is not None:
            if pan_mode == "search" and pan_angle != prev_angle:
                # Force each new search step (bypass rate limits).
                servo_pub.publish_angle(int(pan_angle), force=True)
            elif pan_mode == "lock" and pan_angle != prev_angle:
                servo_pub.publish_angle(int(pan_angle), force=False)

        # --- Publish angle to tachometer ESP8266 (needle tracks face position) ---
        if tacho_pub is not None:
            # is_locked = True only when a KNOWN (non-Unknown) face is locked
            is_locked = (
                pan_mode == "lock"
                and track_label not in ("none", "Unknown", None)
            )
            tacho_pub.update(
                face_cx=face_cx,
                face_cy=face_cy,
                frame_width=w,
                frame_height=h,
                is_locked=is_locked,
                axis=track_axis,
            )

        # Big lock / search status
        if pan_mode == "lock":
            status = f"LOCKED: {track_label}"
            status_color = (0, 255, 0)
        elif pan_mode == "search":
            status = "SEARCHING..."
            status_color = (0, 165, 255)
        elif pan_mode == "hold":
            status = "HOLD (lost)"
            status_color = (0, 255, 255)
        else:
            status = "IDLE"
            status_color = (200, 200, 200)
        cv2.putText(vis, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        header = f"cam={args.cam} IDs={len(matcher._names)} thr(dist)={matcher.dist_thresh:.2f}"
        if fps is not None:
            header += f" fps={fps:.1f}"
        header += f" pan={pan_angle} mode={pan_mode}"
        header += f" mqtt={'ON' if mqtt_enabled else 'OFF'}"
        header += f" search={'ON' if search_enabled else 'OFF'}"
        header += f" axis={track_axis.upper()}"
        if invert_pan:
            header += " inv"
        cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("recognize_new", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            matcher.reload_from(db_path)
            print(f"[recognize] reloaded DB: {len(matcher._names)} identities")
        elif key in (ord("+"), ord("=")):
            matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
            print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
        elif key == ord("-"):
            matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
            print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"[recognize] debug overlay: {'ON' if show_debug else 'OFF'}")
        elif key == ord("m"):
            mqtt_enabled = not mqtt_enabled
            print(f"[recognize] MQTT pan: {'ON' if mqtt_enabled else 'OFF'}")
        elif key == ord("i"):
            invert_pan = not invert_pan
            pan.set_invert(invert_pan)
            print(f"[recognize] pan invert: {'ON' if invert_pan else 'OFF'}")
        elif key == ord("s"):
            search_enabled = not search_enabled
            print(f"[recognize] search-when-lost: {'ON' if search_enabled else 'OFF'}")
        elif key == ord("f"):
            mirror = not mirror
            print(f"[recognize] mirror flip (horizontal): {'ON' if mirror else 'OFF'}")
        elif key == ord("v"):
            track_axis = "horizontal" if track_axis == "vertical" else "vertical"
            print(f"[recognize] tracking axis toggled to: {track_axis.upper()}")

    if tacho_pub is not None:
        tacho_pub.close()
    if servo_pub is not None:
        servo_pub.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
