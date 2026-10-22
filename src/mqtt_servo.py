# src/mqtt_servo.py
"""
Publish face recognition status to the ESP8266 tachometer over MQTT.

Broker/topic must match firmware:
  10.12.73.141:1883
  face/status  ("1" = recognised, "0" = not recognised)
  tacho/state  (LWT topic)
"""
from __future__ import annotations

import time
from typing import Optional

try:
    import paho.mqtt.client as mqtt
except Exception as e:  # pragma: no cover
    mqtt = None
    _MQTT_IMPORT_ERROR = e


DEFAULT_BROKER = "10.12.73.141"
DEFAULT_PORT = 1883
DEFAULT_TOPIC = "face/status"

MIN_ANGLE = 20
MAX_ANGLE = 160
CENTER_ANGLE = 90


class PanController:
    """
    Face present  -> LOCK (gentle center follow)
    Face missing  -> SEARCH (active left/right sweep), same style as before
    """

    def __init__(
        self,
        min_angle: int = MIN_ANGLE,
        max_angle: int = MAX_ANGLE,
        center_angle: int = CENTER_ANGLE,
        deadzone_frac: float = 0.07,
        track_gain: float = 22.0,
        max_step_deg: float = 3.0,
        lost_before_search_s: float = 0.45,
        search_step_deg: float = 5.0,
        search_period_s: float = 0.10,
        invert: bool = False,
    ):
        self.min_angle = int(min_angle)
        self.max_angle = int(max_angle)
        self.center_angle = int(center_angle)
        self.deadzone_frac = float(deadzone_frac)
        self.track_gain = float(track_gain)
        self.max_step_deg = float(max_step_deg)
        self.lost_before_search_s = float(lost_before_search_s)
        self.search_step_deg = float(search_step_deg)
        self.search_period_s = float(search_period_s)
        self.invert = bool(invert)

        self.angle = float(center_angle)
        self.mode = "idle"  # idle | lock | hold | search
        self.locked = False
        self._search_dir = 1.0
        self._last_search_t = 0.0
        self._last_face_t = 0.0
        self._ever_tracked = False
        self._last_mode = "idle"

    def set_invert(self, invert: bool) -> None:
        self.invert = bool(invert)

    def _nudge_toward_face(self, face_cx: float, frame_width: int) -> None:
        err = (float(face_cx) / float(frame_width)) - 0.5
        if abs(err) < self.deadzone_frac:
            return
        if self.invert:
            err = -err
        delta = max(-self.max_step_deg, min(self.max_step_deg, self.track_gain * err))
        self.angle = max(self.min_angle, min(self.max_angle, self.angle + delta))

    def update(
        self,
        face_cx: Optional[float],
        frame_width: int,
        now: Optional[float] = None,
        allow_search: bool = True,
    ) -> tuple[int, str]:
        now = time.time() if now is None else float(now)
        face_seen = face_cx is not None and frame_width > 1

        if face_seen:
            if self.mode == "search":
                print(f"[pan] face found — LOCK @ {int(round(self.angle))}")
            self._ever_tracked = True
            self._last_face_t = now
            self.locked = True
            self.mode = "lock"
            self._nudge_toward_face(float(face_cx), frame_width)
            self._emit_mode_change()
            return int(round(self.angle)), self.mode

        # No face
        self.locked = False
        if not self._ever_tracked:
            self.mode = "idle"
            self._emit_mode_change()
            return int(round(self.angle)), self.mode

        lost_for = now - self._last_face_t
        if lost_for < self.lost_before_search_s or not allow_search:
            self.mode = "hold"
            self._emit_mode_change()
            return int(round(self.angle)), self.mode

        # Active search sweep (same idea as the earlier working sweep)
        self.mode = "search"
        if (now - self._last_search_t) >= self.search_period_s:
            self._last_search_t = now
            self.angle += self._search_dir * self.search_step_deg
            if self.angle >= self.max_angle:
                self.angle = float(self.max_angle)
                self._search_dir = -1.0
            elif self.angle <= self.min_angle:
                self.angle = float(self.min_angle)
                self._search_dir = 1.0
        self._emit_mode_change()
        return int(round(self.angle)), self.mode

    def _emit_mode_change(self) -> None:
        if self.mode != self._last_mode:
            print(f"[pan] mode: {self._last_mode} -> {self.mode} angle={int(round(self.angle))}")
            self._last_mode = self.mode


class ServoPanPublisher:
    def __init__(
        self,
        broker: str = DEFAULT_BROKER,
        port: int = DEFAULT_PORT,
        topic: str = DEFAULT_TOPIC,
        client_id: str = "pc-face-recognition-pan",
        min_publish_interval_s: float = 0.08,
        min_angle_delta: int = 1,
    ):
        if mqtt is None:
            raise RuntimeError(
                f"paho-mqtt is not installed. Run: pip install paho-mqtt\nImport error: {_MQTT_IMPORT_ERROR}"
            )

        self.broker = broker
        self.port = int(port)
        self.topic = topic
        self.min_publish_interval_s = float(min_publish_interval_s)
        self.min_angle_delta = int(min_angle_delta)

        self._last_publish_t = 0.0
        self._last_angle: Optional[int] = None
        self._connected = False

        if hasattr(mqtt, "CallbackAPIVersion"):
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
                client_id=client_id,
            )
        else:
            self._client = mqtt.Client(client_id=client_id)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = rc == 0
        if self._connected:
            print(f"[mqtt] connected {self.broker}:{self.port} topic={self.topic}")
        else:
            print(f"[mqtt] connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print(f"[mqtt] disconnected rc={rc}")

    def connect(self) -> None:
        self._client.connect(self.broker, self.port, keepalive=30)
        self._client.loop_start()
        deadline = time.time() + 3.0
        while time.time() < deadline and not self._connected:
            time.sleep(0.05)
        if not self._connected:
            print("[mqtt] warning: not connected yet; will keep trying in background")

    def close(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

    def publish_angle(self, angle: int, force: bool = False) -> bool:
        angle = int(max(MIN_ANGLE, min(MAX_ANGLE, angle)))
        now = time.time()

        if not force:
            if self._last_angle is not None and abs(angle - self._last_angle) < self.min_angle_delta:
                return False
            if (now - self._last_publish_t) < self.min_publish_interval_s:
                return False

        payload = str(angle)
        info = self._client.publish(self.topic, payload, qos=0)
        self._last_publish_t = now
        self._last_angle = angle
        ok = info.rc == mqtt.MQTT_ERR_SUCCESS
        if ok:
            print(f"[mqtt] publish {self.topic} -> {payload}")
        else:
            print(f"[mqtt] publish failed rc={info.rc}")
        return ok
