# src/detect.py
"""Minimal YuNet face-box demo. Run: python -m src.detect"""
from __future__ import annotations

import cv2

from .yunet_5pt import YuNet5ptDetector


def main():
    det = YuNet5ptDetector(smooth_alpha=0.50, debug=False)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")

    print("YuNet face detect. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame, max_faces=5)
        for f in faces:
            cv2.rectangle(frame, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{f.score:.2f}",
                (f.x1, max(0, f.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Face Detection (YuNet)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
