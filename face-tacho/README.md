# face-tacho

PC face recognition moves a **tachometer needle** (28BYJ-48 stepper + ULN2003)
through **MQTT** and a **Wemos D1 mini (ESP8266)**.

```text
Camera → Python (ArcFace DB) → Mosquitto on PC → ESP8266 → ULN2003 → stepper → needle
```

- Recognised person → needle goes to **HIGH** (right / +90°)
- Unknown / gone → needle returns to **ZERO** (left / -90°)

---

## What do -90 / +180 / -90 mean?

Your dial is marked like a gauge from **left to right**:

```text
   LEFT (-90°)          CENTER (0°)          RIGHT (+90°)
        \                    |                    /
         \                   |                   /
          ●──────────────────●──────────────────●
        ZERO                                  HIGH
   (not recognised)                      (recognised)
```

| Concept | Meaning |
|--------|---------|
| **-90°** | Needle resting on the **left** (zero / not recognised) at power-up |
| **+180°** | How far the motor turns to reach the **right** mark (from -90 to +90) |
| Return **-180°** | Same distance back to zero |

So the needle does **not** spin a full circle. It sweeps **180°** of the dial.

In code (`esp/tacho.ino`):

```cpp
SWEEP_DEG = 180.0;
HIGH_POS  = 4096 * 180 / 360;  // = 2048 half-steps
```

If your physical dial is wider (e.g. 240°), change `SWEEP_DEG` to `240.0`.

---

## Wiring (Wemos D1 mini ↔ ULN2003)

| ULN2003 board | Wemos D1 mini |
|---------------|---------------|
| IN1 | D1 (GPIO5) |
| IN2 | D2 (GPIO4) |
| IN3 | D5 (GPIO14) |
| IN4 | D6 (GPIO12) |
| GND | G |
| + (motor power) | **5 V** (not 3.3 V) |

- Share **GND** between ESP, ULN2003, and motor supply.
- Prefer an external **5 V** supply for the motor if the USB port browns out.
- Do **not** use D8 / D3 for motor pins (boot pins).

---

## Folder layout

```text
face-tacho/
├── pc/
│   ├── config.py          # BROKER_IP, camera, threshold, debounce
│   ├── mqtt_client.py     # paho-mqtt 2.x
│   ├── face_tracker.py    # camera + ArcFace + publish face/status
│   ├── timer_pub.py       # MQTT test without camera
│   ├── requirements.txt
│   └── known_faces/       # optional notes / photos (DB is parent data/db)
├── broker/
│   └── mosquitto.conf
├── esp/
│   └── tacho.ino
└── README.md
```

Recognition model: **parent project** YuNet 5pt + ArcFace ONNX  
(`../data/db/face_db.npz`, `../models/embedder_arcface.onnx`).

---

## 1. Find your PC IP (Windows)

```powershell
ipconfig
```

Use the **IPv4 Address** under your Wi-Fi adapter (example `192.168.1.10`).

Put that IP in:

- `pc/config.py` → `BROKER_IP`
- `esp/tacho.ino` → `MQTT_BROKER`

ESP must use the **PC LAN IP**, never `127.0.0.1` / `localhost`.

---

## 2. Install and start Mosquitto (Windows)

1. Install [Mosquitto](https://mosquitto.org/download/).
2. Copy or point Mosquitto at `broker/mosquitto.conf`.
3. Start (example):

```powershell
mosquitto -c "C:\Users\user\OneDrive\Desktop\Robotics Projects\Face-Recognition-5pt\face-tacho\broker\mosquitto.conf" -v
```

4. Open firewall port **1883** (once):

```powershell
netsh advfirewall firewall add rule name="Mosquitto 1883" dir=in action=allow protocol=TCP localport=1883
```

---

## 3. Flash the ESP

1. Arduino IDE → Board: **LOLIN(WEMOS) D1 R2 & mini**
2. Install libraries: **PubSubClient**, **AccelStepper**
3. Open `esp/tacho.ino`
4. Set `WIFI_SSID`, `WIFI_PASS`, `MQTT_BROKER`
5. Upload, Serial Monitor **115200**

You should see Wi-Fi + MQTT connect. Needle assumes **ZERO** at power-up (left / -90).

---

## 4. PC Python setup

```powershell
cd "C:\Users\user\OneDrive\Desktop\Robotics Projects\Face-Recognition-5pt\face-tacho\pc"
..\..\.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Edit `config.py`: `BROKER_IP`, `CAMERA_INDEX` (1 for USB cam).

Enroll faces in the **parent** project if needed:

```powershell
cd ..\..
.\.venv\Scripts\python.exe -m src.enroll --cam 1
```

---

## Test plan (do in this order)

### a) Broker only

Terminal 1 — subscribe:

```powershell
mosquitto_sub -h 127.0.0.1 -t "face/#" -t "tacho/#" -v
```

Terminal 2 — publish:

```powershell
mosquitto_pub -h 127.0.0.1 -t face/status -m 1 -r
mosquitto_pub -h 127.0.0.1 -t face/status -m 0 -r
```

### b) ESP + stepper alone (no camera)

With ESP online, from the PC:

```powershell
mosquitto_pub -h 192.168.1.10 -t face/status -m 1 -r
mosquitto_pub -h 192.168.1.10 -t face/status -m 0 -r
```

Needle should go HIGH then back to ZERO. Watch ESP Serial.

### c) Python timer (no camera)

```powershell
cd face-tacho\pc
python timer_pub.py
```

Needle should toggle every ~4 seconds.

### d) Recognition alone (prints)

Run `face_tracker.py` and confirm green box + name; MQTT lines print `face/status -> 1/0`.

### e) Full integration

1. Mosquitto running  
2. ESP powered and `tacho/state = online`  
3. `python face_tracker.py`  
4. Show a known face → needle HIGH  
5. Leave frame → needle ZERO  

---

## MQTT topics

| Topic | Direction | Payload |
|-------|-----------|---------|
| `face/status` | PC → ESP | `"1"` / `"0"` (retained) |
| `face/name` | PC → ESP | person name (optional) |
| `tacho/state` | ESP → PC | `"online"` / `"offline"` (LWT retained) |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| ESP never connects MQTT | Broker only on localhost / wrong IP / firewall | Use PC LAN IP in `.ino`; open port 1883; `listener 1883` in conf |
| Motor vibrates, no turn | Wrong pin order | Confirm IN1–IN4 wiring; AccelStepper order is D1,D5,D2,D6 |
| ESP resets when motor starts | Weak 5 V / USB brownout | Separate 5 V supply for ULN2003 `+`, common GND |
| Needle drifts over time | Open-loop stepper | Power-cycle to re-zero; keep `setCurrentPosition(0)` at boot with needle at left |
| Wrong sweep distance | Dial not 180° | Change `SWEEP_DEG` in `tacho.ino` |

---

## Optional: confidence-scaled needle

In `pc/config.py` set `USE_CONFIDENCE_SCALE = True`.  
Then extend `tacho.ino` to subscribe to `face/confidence` (`0.00`–`1.00`) and:

```cpp
stepper.moveTo((long)(HIGH_POS * confidence));
```

Keep binary `face/status` for the basic assignment unless your teacher asks for proportional mode.
