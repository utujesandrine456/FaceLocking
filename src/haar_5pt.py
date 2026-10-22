# src/haar_5pt.py
"""
Compatibility shim — Haar was replaced by YuNet.

Prefer: `from .yunet_5pt import YuNet5ptDetector, align_face_5pt`
"""
from __future__ import annotations

from .yunet_5pt import (  # noqa: F401
    FaceKpsBox,
    Haar5ptDetector,
    YuNet5ptDetector,
    YuNetFaceDetector,
    align_face_5pt,
)

__all__ = [
    "FaceKpsBox",
    "Haar5ptDetector",
    "YuNet5ptDetector",
    "YuNetFaceDetector",
    "align_face_5pt",
]


def main():
    from .yunet_5pt import main as yunet_main

    yunet_main()


if __name__ == "__main__":
    main()
