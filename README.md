# face-recognition-5pt

A CPU-only face recognition + **whole-person tracking** pipeline:  
YuNet detection (box + 5-point landmarks) + similarity-transform alignment + ArcFace ONNX embeddings  
**+ YOLOX-Nano body detection + Three-State temporal identity continuity.**

See the accompanying book for full explanations.

---

## Quick Start

1. Install Python 3.11+ (3.11 recommended).
2. `py -3.11 -m venv .venv`
3. `.venv\Scripts\Activate.ps1`  (Windows PowerShell)
4. `pip install -r requirements.txt`
5. Ensure `models/face_detection_yunet_2023mar.onnx` is present (OpenCV Zoo YuNet).

### Validation Steps
```
python -m src.camera        # validate webcam
python -m src.detect        # validate face box
python -m src.landmarks     # validate 5 points
python -m src.align         # validate alignment
python -m src.embed         # validate ArcFace embeddings
```

### Enrollment
```
python -m src.enroll        # capture face samples per identity
```

### Training the Face Database
```
python -m src.train_db      # compile all enrolled faces → data/db/face_db.npz
```
Outputs:
- `data/db/face_db.npz` — 512-d L2-normalized ArcFace mean embeddings per person
- `data/db/face_db.json` — metadata + inter-identity distance matrix

### Recognition & Tracking

**Classic single-face recognition** (original):
```
python -m src.recognize     # YuNet + ArcFace
```

**NEW — Whole-Person Tracking + Three-State Identity** (recommended):
```
python -m src.track_and_recognize
python -m src.track_and_recognize --cam 0      # laptop webcam
python -m src.track_and_recognize --search     # enable search-on-lost at startup
```

#### Three-State Identity Model

| State | Meaning | Display |
|-------|---------|---------|
| **CONFIRMED** | Face actively detected and verified | 🟢 `CONFIRMED: Alice (91%)` |
| **TEMPORARY** | Body tracked, face temporarily absent (head turn / occlusion). Confidence decays exponentially: `c(t) = c₀ · e^(−λt)` | 🟡 `TEMPORARY: Alice? (64%, dt=1.2s)` |
| **UNKNOWN** | Confidence exhausted or no enrolled match | 🔴 `UNKNOWN` |

#### Autonomous Search
When a tracked person disappears, the camera sweeps left/right automatically (simulated pan angle) to find them again. Toggle with `s` key.

#### Keys in `track_and_recognize`
| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reload face DB from disk |
| `+`/`-` | Adjust cosine distance threshold |
| `d` | Toggle debug overlay (centroid trails) |
| `s` | Toggle autonomous search-when-lost |
| `f` | Toggle mirror flip |
| `m` | Toggle MQTT angle publish |

---

## New Module Architecture

```
src/
├── train_db.py          ← Batch DB training from enroll/ crops
├── yolox_detector.py    ← YOLOX-Nano body detector (CPU, ONNX)
├── identity_state.py    ← Three-State model + exponential decay
├── body_tracker.py      ← Multi-person IoU tracker + face-body spatial association
├── target_centering.py  ← Target centering controller + autonomous search HUD
└── track_and_recognize.py ← Main integrated pipeline (run this)
```

---

## Embedded (ESP8266 servo pan over MQTT)

Hardware firmware lives in `firmware/esp8266_servo_pan/`.

1. Wire servo signal to D1 (GPIO5). Power servo from external 5V. Common GND with ESP8266.
2. Flash the sketch from Arduino IDE (PubSubClient + ESP8266 Servo).
3. Open Serial Monitor at 115200: WiFi + MQTT should connect; servo self-test runs.
4. On the PC (venv active):
   - `pip install -r requirements.txt`
   - `python -m src.servo_test`  (moves servo center/left/right without camera)
   - `python -m src.track_and_recognize`   (tracks face X and publishes pan angles)
5. Keys in track_and_recognize: `m` MQTT on/off, `s` search-when-lost on/off.

MQTT: `broker.benax.rw:1883` / `10.12.73.141:1883`  topic `face/status` (payload = angle int).
