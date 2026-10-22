# src/yunet_5pt.py
"""
YuNet face detection + ArcFace 5-point alignment.

YuNet (cv2.FaceDetectorYN) is a single-pass detector that returns a bounding
box, 5 facial landmarks, and a confidence score — no Haar cascade and no
MediaPipe FaceMesh required for detection/alignment.

Run: python -m src.yunet_5pt
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_YUNET_MODEL = Path("models/face_detection_yunet_2023mar.onnx")


@dataclass
class FaceKpsBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 — left_eye, right_eye, nose, mouth_left, mouth_right


def _estimate_norm_5pt(kps_5x2: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    k = kps_5x2.astype(np.float32)
    dst = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    out_w, out_h = int(out_size[0]), int(out_size[1])
    if (out_w, out_h) != (112, 112):
        sx = out_w / 112.0
        sy = out_h / 112.0
        dst = dst * np.array([sx, sy], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(k, dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(
            np.array([k[0], k[1], k[2]], dtype=np.float32),
            np.array([dst[0], dst[1], dst[2]], dtype=np.float32),
        )
    return M.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112),
) -> Tuple[np.ndarray, np.ndarray]:
    M = _estimate_norm_5pt(kps_5x2, out_size=out_size)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, M


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def resolve_yunet_model_path(model_path: Optional[str] = None) -> str:
    path = Path(model_path) if model_path else DEFAULT_YUNET_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"YuNet model not found: {path}\n"
            "Download from:\n"
            "  https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx\n"
            f"and place it at: {DEFAULT_YUNET_MODEL}"
        )
    return str(path)


class YuNet5ptDetector:
    """
    Single-pass YuNet face detector with ArcFace-ordered 5-point landmarks.

    Landmark order from YuNet: right_eye, left_eye, nose_tip, mouth_right, mouth_left
    Reordered to ArcFace:       left_eye, right_eye, nose_tip, mouth_left, mouth_right
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        score_thresh: float = 0.60,
        nms_thresh: float = 0.30,
        smooth_alpha: float = 0.65,
        debug: bool = False,
        # Legacy Haar kwargs accepted but ignored (drop-in compatibility)
        haar_xml: Optional[str] = None,
        min_size: Optional[Tuple[int, int]] = None,
    ):
        self.debug = bool(debug)
        self.smooth_alpha = float(smooth_alpha)
        self.score_thresh = float(score_thresh)

        resolved = resolve_yunet_model_path(model_path)
        self._detector = cv2.FaceDetectorYN.create(
            model=resolved,
            config="",
            input_size=(320, 320),
            score_threshold=score_thresh,
            nms_threshold=nms_thresh,
            top_k=5,
        )
        self._last_size = (0, 0)

        self._prev_kps: Dict[int, np.ndarray] = {}
        self._prev_box: Dict[int, np.ndarray] = {}
        self._slot_centers: Dict[int, Tuple[float, float]] = {}
        self._next_slot: int = 0

        if self.debug:
            print(f"[yunet] loaded {resolved}")

    def _match_slot(self, cx: float, cy: float, max_dist: float = 90.0) -> int:
        best_slot, best_d = -1, max_dist
        for sid, (sx, sy) in self._slot_centers.items():
            d = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_slot = d, sid
        if best_slot >= 0:
            sx, sy = self._slot_centers[best_slot]
            self._slot_centers[best_slot] = (0.7 * sx + 0.3 * cx, 0.7 * sy + 0.3 * cy)
            return best_slot
        sid = self._next_slot
        self._next_slot += 1
        self._slot_centers[sid] = (cx, cy)
        return sid

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceKpsBox]:
        H, W = frame_bgr.shape[:2]

        if (W, H) != self._last_size:
            self._detector.setInputSize((W, H))
            self._last_size = (W, H)

        _, raw = self._detector.detect(frame_bgr)
        if raw is None or len(raw) == 0:
            self._prev_kps.clear()
            self._prev_box.clear()
            self._slot_centers.clear()
            return []

        raw = raw[np.argsort(raw[:, 14])[::-1]][:max_faces]

        results: List[FaceKpsBox] = []
        active_slots: set = set()

        for det in raw:
            bx, by, bw, bh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            score = float(det[14])

            # YuNet: [right_eye, left_eye, nose_tip, mouth_right, mouth_left]
            raw_kps = det[4:14].reshape(5, 2).astype(np.float32)
            # ArcFace: [left_eye, right_eye, nose_tip, mouth_left, mouth_right]
            kps = raw_kps[[1, 0, 2, 4, 3]].copy()

            eye_dist = float(np.linalg.norm(kps[0] - kps[1]))
            if eye_dist < max(8.0, 0.10 * bw):
                continue

            cx = float(bx + bw * 0.5)
            cy = float(by + bh * 0.5)
            slot = self._match_slot(cx, cy)
            active_slots.add(slot)

            bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
            kps_s = _ema(self._prev_kps.get(slot), kps, self.smooth_alpha)
            bb_s = _ema(self._prev_box.get(slot), bb, self.smooth_alpha)
            self._prev_kps[slot] = kps_s.copy()
            self._prev_box[slot] = bb_s.copy()

            x1, y1, x2, y2 = _clip_xyxy(bb_s[0], bb_s[1], bb_s[2], bb_s[3], W, H)
            results.append(
                FaceKpsBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    kps=kps_s.astype(np.float32),
                )
            )

        for sid in [s for s in list(self._slot_centers) if s not in active_slots]:
            self._slot_centers.pop(sid, None)
            self._prev_kps.pop(sid, None)
            self._prev_box.pop(sid, None)

        return results


# Backward-compatible names used across the project
YuNetFaceDetector = YuNet5ptDetector
Haar5ptDetector = YuNet5ptDetector


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")

    det = YuNet5ptDetector(smooth_alpha=0.65, debug=True)
    print("YuNet 5pt detect. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame, max_faces=5)
        vis = frame.copy()

        if faces:
            for f in faces:
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.putText(
                    vis,
                    f"{f.score:.2f}",
                    (f.x1, max(0, f.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
        else:
            cv2.putText(vis, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("yunet_5pt", vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
