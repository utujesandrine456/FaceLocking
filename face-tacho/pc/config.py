"""
pc/config.py — single place for PC-side settings.
Edit BROKER_IP and CAMERA_INDEX for your setup.
"""

# ---- MQTT broker (Mosquitto on this PC) ----
# Python runs ON this PC → use localhost (fast, reliable):
BROKER_IP = "127.0.0.1"
BROKER_PORT = 1883

# ESP8266 must use your Wi-Fi IPv4 (from ipconfig), NOT 127.0.0.1 and NOT 172.x (WSL).
# Right now your Wi-Fi IP is: 10.12.73.178
# Put that same address in esp/tacho.ino as MQTT_BROKER.
WIFI_LAN_IP_FOR_ESP = "10.12.73.178"

TOPIC_STATUS = "face/status"   # "1" = recognised, "0" = not
TOPIC_NAME = "face/name"       # optional person name
TOPIC_TACHO = "tacho/state"    # ESP LWT online/offline

MQTT_CLIENT_ID = "pc-face-tacho"
MQTT_KEEPALIVE = 30

# ---- Camera ----
CAMERA_INDEX = 1               # 0 = laptop, 1/2 = USB (try --list in parent project)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ---- Recognition (uses parent Face-Recognition-5pt ArcFace DB) ----
# Paths relative to face-tacho/pc/ when run as: python face_tracker.py
# from face-tacho/pc, or set absolute paths if needed.
PROJECT_ROOT = r"C:\Users\user\OneDrive\Desktop\Robotics Projects\Face-Recognition-5pt"
FACE_DB_NPZ = PROJECT_ROOT + r"\data\db\face_db.npz"
EMBEDDER_ONNX = PROJECT_ROOT + r"\models\embedder_arcface.onnx"
YUNET_ONNX = PROJECT_ROOT + r"\models\face_detection_yunet_2023mar.onnx"

# Cosine distance threshold (lower = stricter). ~0.40–0.45 typical.
DIST_THRESH = 0.45

# Debounce: need N consecutive hits/misses before publishing a new state
DEBOUNCE_FRAMES = 5

# Preview window
WINDOW_NAME = "face-tacho"

# Optional: scale needle by confidence (0..1) instead of binary 1/0
# ESP must subscribe to face/confidence for this (see tacho.ino optional block)
USE_CONFIDENCE_SCALE = False
TOPIC_CONFIDENCE = "face/confidence"  # payload "0.00" .. "1.00"
