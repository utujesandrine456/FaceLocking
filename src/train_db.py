# src/train_db.py
"""
Batch Face Database Training / Enrollment:
Reads all enrolled identity folders from `data/enroll/`,
extracts ArcFace (512-d) embeddings using `models/embedder_arcface.onnx`,
computes the L2-normalized mean template for each identity,
and saves the compiled database to `data/db/face_db.npz` and `data/db/face_db.json`.

Also computes inter-identity separation distances and quality metrics
to guarantee high discrimination accuracy.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX, resolve_embedder_model_path


def load_crops_for_person(person_dir: Path) -> List[np.ndarray]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([p for p in person_dir.iterdir() if p.suffix.lower() in valid_exts])
    images = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
    return images


def compute_mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Cannot compute mean of empty embedding list")
    stacked = np.stack([e.reshape(-1).astype(np.float32) for e in embeddings], axis=0)
    mean_vec = np.mean(stacked, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm < 1e-12:
        norm = 1e-12
    return (mean_vec / norm).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    sim = float(np.dot(a.reshape(-1), b.reshape(-1)))
    return 1.0 - sim


def train_database(
    enroll_dir: Path = Path("data/enroll"),
    db_npz_path: Path = Path("data/db/face_db.npz"),
    db_json_path: Path = Path("data/db/face_db.json"),
    model_path: str = "models/embedder_arcface.onnx",
) -> Dict[str, np.ndarray]:
    model_path = resolve_embedder_model_path(model_path)
    embedder = ArcFaceEmbedderONNX(model_path=model_path, input_size=(112, 112), debug=False)

    person_dirs = sorted([d for d in enroll_dir.iterdir() if d.is_dir()])
    if not person_dirs:
        raise RuntimeError(f"No identity directories found in {enroll_dir}")

    db: Dict[str, np.ndarray] = {}
    stats: Dict[str, int] = {}
    all_raw_embeddings: Dict[str, List[np.ndarray]] = {}

    print(f"\n==================================================")
    print(f" TRAINING FACE EMBEDDING DATABASE")
    print(f" Source directory: {enroll_dir}")
    print(f" ArcFace Model   : {model_path}")
    print(f"==================================================")

    total_samples = 0
    for pdir in person_dirs:
        name = pdir.name
        crops = load_crops_for_person(pdir)
        if not crops:
            print(f"[-] {name}: 0 images found (skipping)")
            continue

        embs = []
        for img in crops:
            res = embedder.embed(img)
            embs.append(res.embedding)

        mean_emb = compute_mean_embedding(embs)
        db[name] = mean_emb
        stats[name] = len(embs)
        all_raw_embeddings[name] = embs
        total_samples += len(embs)
        print(f"[+] {name:15s}: {len(embs):3d} samples -> mean 512-d embedding generated")

    if not db:
        raise RuntimeError("No valid face samples were processed!")

    # Inter-identity distance analysis
    names = sorted(db.keys())
    print("\n--- Inter-Identity Distance Matrix (Cosine Distance: 0=identical, 1=orthogonal) ---")
    header = f"{'':15s}" + "".join([f"{n[:10]:>12s}" for n in names])
    print(header)
    for i, n1 in enumerate(names):
        row = f"{n1:15s}"
        for j, n2 in enumerate(names):
            d = cosine_distance(db[n1], db[n2])
            row += f"{d:12.4f}"
        print(row)

    # Within-identity cohesion (mean distance of samples to their class centroid)
    print("\n--- Intra-Identity Cohesion (Variance to Centroid) ---")
    for name in names:
        embs = all_raw_embeddings[name]
        dists = [cosine_distance(e, db[name]) for e in embs]
        mean_d = float(np.mean(dists))
        max_d = float(np.max(dists))
        print(f"  {name:15s}: mean distance={mean_d:.4f}, max distance={max_d:.4f} (samples={len(embs)})")

    # Save to disk
    db_npz_path.parent.mkdir(parents=True, exist_ok=True)
    db_json_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(db_npz_path, **{k: v.astype(np.float32) for k, v in db.items()})

    meta = {
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_dim": 512,
        "names": names,
        "sample_counts": stats,
        "samples_total_used": total_samples,
        "model_path": str(model_path),
        "note": "L2-normalized ArcFace mean embeddings. Metric: cosine distance/similarity.",
    }
    db_json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[OK] Successfully trained and saved database:")
    print(f"     NPZ : {db_npz_path} ({len(db)} identities)")
    print(f"     JSON: {db_json_path}")
    print(f"==================================================\n")

    return db


def main():
    parser = argparse.ArgumentParser(description="Train face database from enrollment images")
    parser.add_argument("--enroll-dir", type=Path, default=Path("data/enroll"))
    parser.add_argument("--npz", type=Path, default=Path("data/db/face_db.npz"))
    parser.add_argument("--json", type=Path, default=Path("data/db/face_db.json"))
    args = parser.parse_args()

    train_database(
        enroll_dir=args.enroll_dir,
        db_npz_path=args.npz,
        db_json_path=args.json,
    )


if __name__ == "__main__":
    main()
