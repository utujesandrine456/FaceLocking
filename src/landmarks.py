# src/landmarks.py
"""
Minimal pipeline:
camera -> YuNet face box + 5 keypoints -> draw

Run: python -m src.landmarks
Keys: q : quit
"""
from __future__ import annotations

import cv2

from .yunet_5pt import YuNet5ptDetector


def main():
    det = YuNet5ptDetector(smooth_alpha=0.50, debug=False)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")

    print("YuNet 5pt landmarks. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame, max_faces=5)
        for f in faces:
            cv2.rectangle(frame, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (px, py) in f.kps.astype(int):
                cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)
            cv2.putText(
                frame,
                "5pt",
                (f.x1, max(0, f.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if not faces:
            cv2.putText(frame, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("5pt Landmarks (YuNet)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
