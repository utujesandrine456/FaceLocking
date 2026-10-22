# src/yolox_detector.py
"""
YOLOX-Nano CPU-optimized Object Detector for Person Detection.
Loads `models/yolox_nano.onnx` using onnxruntime CPUExecutionProvider.
Performs letterbox preprocessing, inference, and NMS for person bounding boxes.
Includes a graceful fallback (HOG / MediaPipe) if the ONNX model is unavailable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_YOLOX_MODEL = Path("models/yolox_nano.onnx")


@dataclass
class BodyDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int = 0  # 0 = person

    @property
    def box_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return float(self.width * self.height)


class YOLOXPersonDetector:
    """
    High-speed, CPU-friendly YOLOX detector dedicated to person detection.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_YOLOX_MODEL,
        input_size: Tuple[int, int] = (416, 416),
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.45,
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "images"
        self._fallback_hog: Optional[cv2.HOGDescriptor] = None

        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
        else:
            print(f"[YOLOX] Warning: {self.model_path} not found. Using OpenCV HOG person detector fallback.")
            self._init_fallback()

    def _init_fallback(self) -> None:
        self._fallback_hog = cv2.HOGDescriptor()
        self._fallback_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        target_h, target_w = self.input_size
        padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 114

        r = min(target_h / img_bgr.shape[0], target_w / img_bgr.shape[1])
        new_w = int(img_bgr.shape[1] * r)
        new_h = int(img_bgr.shape[0] * r)
        resized_img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img[:new_h, :new_w] = resized_img

        # BGR (H, W, C) -> (C, H, W)
        blob = padded_img.transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob[None, ...], dtype=np.float32)
        return blob, r

    def _postprocess(
        self, outputs: np.ndarray, r: float, orig_w: int, orig_h: int
    ) -> List[BodyDet]:
        # outputs shape: (1, 3549, 85)
        preds = outputs[0]
        # boxes: cx, cy, w, h
        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]

        obj_conf = preds[:, 4]
        # COCO class 0 is person
        person_cls_conf = preds[:, 5]
        person_scores = obj_conf * person_cls_conf

        mask = person_scores >= self.conf_thresh
        if not np.any(mask):
            return []

        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]
        scores = person_scores[mask]

        # Convert cx, cy, w, h to x1, y1, x2, y2 on original frame
        x1 = (cx - w / 2.0) / r
        y1 = (cy - h / 2.0) / r
        x2 = (cx + w / 2.0) / r
        y2 = (cy + h / 2.0) / r

        # Clip to image boundaries
        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        boxes_list = []
        for i in range(len(scores)):
            bx = int(x1[i])
            by = int(y1[i])
            bw = int(max(1, x2[i] - x1[i]))
            bh = int(max(1, y2[i] - y1[i]))
            boxes_list.append([bx, by, bw, bh])

        indices = cv2.dnn.NMSBoxes(
            boxes_list,
            scores.tolist(),
            score_threshold=self.conf_thresh,
            nms_threshold=self.nms_thresh,
        )

        results: List[BodyDet] = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                bx, by, bw, bh = boxes_list[idx]
                results.append(
                    BodyDet(
                        x1=bx,
                        y1=by,
                        x2=bx + bw,
                        y2=by + bh,
                        score=float(scores[idx]),
                        class_id=0,
                    )
                )

        return results

    def _detect_fallback(self, frame_bgr: np.ndarray) -> List[BodyDet]:
        if self._fallback_hog is None:
            self._init_fallback()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes, weights = self._fallback_hog.detectMultiScale(
            gray, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        results = []
        for i, (x, y, w, h) in enumerate(boxes):
            sc = float(weights[i]) if len(weights) > i else 0.6
            results.append(
                BodyDet(x1=int(x), y1=int(y), x2=int(x + w), y2=int(y + h), score=sc, class_id=0)
            )
        return results

    def detect(self, frame_bgr: np.ndarray) -> List[BodyDet]:
        orig_h, orig_w = frame_bgr.shape[:2]
        if self.session is None:
            return self._detect_fallback(frame_bgr)

        try:
            blob, r = self._preprocess(frame_bgr)
            outputs = self.session.run(None, {self.input_name: blob})[0]
            return self._postprocess(outputs, r, orig_w, orig_h)
        except Exception as e:
            print(f"[YOLOX] Inference error: {e}. Falling back to HOG.")
            return self._detect_fallback(frame_bgr)
