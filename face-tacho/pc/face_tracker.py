"""
pc/face_tracker.py
Camera -> ArcFace recognition (parent project DB) -> MQTT face/status -> ESP tachometer.

Run from this folder:
  cd face-tacho/pc
  python face_tracker.py

Keys: q = quit
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config
from mqtt_client import FaceMqttClient

# Allow importing the parent Face-Recognition-5pt src package
ROOT = Path(config.PROJECT_ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yunet_5pt import YuNet5ptDetector, align_face_5pt  # noqa: E402
from src.embed import ArcFaceEmbedderONNX  # noqa: E402


def load_db(npz_path: str) -> dict:
    p = Path(npz_path)
    if not p.exists():
        print(f"[warn] face DB not found: {p}")
        print("       Enroll people first in the parent project: python -m src.enroll --cam 1")
        return {}
    data = np.load(str(p), allow_pickle=True)
    return {k: np.asarray(data[k], dtype=np.float32).reshape(-1) for k in data.files}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(1.0 - np.dot(a, b))


def match(emb: np.ndarray, db: dict, thresh: float):
    if not db:
        return None, 1.0, 0.0
    best_name, best_dist = None, 1.0
    for name, proto in db.items():
        d = cosine_distance(emb, proto)
        if d < best_dist:
            best_dist = d
            best_name = name
    sim = 1.0 - best_dist
    if best_dist <= thresh:
        return best_name, best_dist, sim
    return None, best_dist, sim


def main():
    # Ensure relative parent assets resolve even when launched from face-tacho/pc
    for path_name, path_val in (
        ("YUNET_ONNX", config.YUNET_ONNX),
        ("EMBEDDER_ONNX", config.EMBEDDER_ONNX),
        ("FACE_DB_NPZ", config.FACE_DB_NPZ),
    ):
        if not Path(path_val).exists():
            raise FileNotFoundError(f"{path_name} not found: {path_val}")

    db = load_db(config.FACE_DB_NPZ)
    print(f"[db] identities: {sorted(db.keys()) or '(none)'}")

    det = YuNet5ptDetector(
        model_path=config.YUNET_ONNX,
        smooth_alpha=0.65,
        debug=False,
    )
    embedder = ArcFaceEmbedderONNX(
        model_path=config.EMBEDDER_ONNX,
        input_size=(112, 112),
        debug=False,
    )

    mqtt = FaceMqttClient()
    mqtt.connect()
    mqtt.publish_status("0", force=True)

    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Camera index {config.CAMERA_INDEX} failed. "
            "Try CAMERA_INDEX = 0 or 2 in config.py"
        )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    # Debounce counters
    hit_streak = 0
    miss_streak = 0
    recognised = False
    current_name = ""

    print(f"[cam] index={config.CAMERA_INDEX}  thresh={config.DIST_THRESH}  debounce={config.DEBOUNCE_FRAMES}")
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[cam] failed to read frame")
                break

            vis = frame.copy()
            faces = det.detect(frame, max_faces=1)
            frame_hit = False
            name = ""
            dist, sim = 1.0, 0.0

            if faces:
                f = faces[0]
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb = embedder.embed(aligned).embedding
                name, dist, sim = match(emb, db, config.DIST_THRESH)
                frame_hit = name is not None

                color = (0, 255, 0) if frame_hit else (0, 0, 255)
                label = name if frame_hit else "Unknown"
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                cv2.putText(
                    vis, f"{label}  dist={dist:.3f}  conf={sim:.2f}",
                    (f.x1, max(20, f.y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                )

            # Debounce state machine
            if frame_hit:
                hit_streak += 1
                miss_streak = 0
                if not recognised and hit_streak >= config.DEBOUNCE_FRAMES:
                    recognised = True
                    current_name = name or ""
                    mqtt.publish_status("1")
                    mqtt.publish_name(current_name)
                    print(f"[state] RECOGNISED -> {current_name}")
            else:
                miss_streak += 1
                hit_streak = 0
                if recognised and miss_streak >= config.DEBOUNCE_FRAMES:
                    recognised = False
                    current_name = ""
                    mqtt.publish_status("0")
                    mqtt.publish_name("")
                    print("[state] NOT recognised")

            if config.USE_CONFIDENCE_SCALE:
                conf = sim if recognised else 0.0
                mqtt.publish_confidence(conf)

            status = f"{'HIT' if recognised else 'MISS'}  name={current_name or '-'}"
            cv2.putText(vis, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.imshow(config.WINDOW_NAME, vis)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[exit] Ctrl+C")
    finally:
        print("[exit] publishing 0 and cleaning up...")
        mqtt.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
