"""
pc/mqtt_client.py — paho-mqtt 2.x wrapper for face/status publishing.
"""
from __future__ import annotations

import time
from typing import Optional

import paho.mqtt.client as mqtt

import config


class FaceMqttClient:
    """Publish face recognition status to the ESP over MQTT."""

    def __init__(
        self,
        broker: str = config.BROKER_IP,
        port: int = config.BROKER_PORT,
        client_id: str = config.MQTT_CLIENT_ID,
    ):
        self.broker = broker
        self.port = int(port)
        self._last_status: Optional[str] = None
        self._connected = False

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        ok = reason_code == 0 or str(reason_code) in ("Success", "0")
        self._connected = bool(ok)
        if self._connected:
            print(f"[mqtt] connected {self.broker}:{self.port}")
        else:
            print(f"[mqtt] connect failed: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self._connected = False
        print(f"[mqtt] disconnected: {reason_code}")

    def connect(self) -> None:
        print(f"[mqtt] connecting to {self.broker}:{self.port} ...")
        try:
            self._client.connect(self.broker, self.port, keepalive=config.MQTT_KEEPALIVE)
            self._client.loop_start()
        except OSError as e:
            raise ConnectionError(
                f"Cannot reach MQTT broker at {self.broker}:{self.port} ({e}).\n"
                f"  - Is Mosquitto running?\n"
                f"  - For Python on this PC use BROKER_IP = '127.0.0.1' in config.py\n"
                f"  - For ESP use your Wi-Fi IP: {getattr(config, 'WIFI_LAN_IP_FOR_ESP', 'ipconfig')}"
            ) from e
        deadline = time.time() + 5.0
        while time.time() < deadline and not self._connected:
            time.sleep(0.05)
        if not self._connected:
            print("[mqtt] warning: not connected yet (will keep trying)")

    def close(self) -> None:
        try:
            self.publish_status("0", force=True)
            time.sleep(0.1)
            self._client.loop_stop()
            self._client.disconnect()
        except Exception as e:
            print(f"[mqtt] close error: {e}")

    def publish_status(self, state: str, force: bool = False) -> bool:
        """
        Publish "1" or "0" to face/status with retain=True.
        Only publishes when the value changes (unless force=True).
        """
        state = "1" if str(state).strip() in ("1", "true", "True", "recognised") else "0"
        if not force and state == self._last_status:
            return False
        info = self._client.publish(config.TOPIC_STATUS, state, qos=1, retain=True)
        self._last_status = state
        print(f"[mqtt] {config.TOPIC_STATUS} -> {state} (rc={info.rc})")
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def publish_name(self, name: str) -> None:
        self._client.publish(config.TOPIC_NAME, name or "", qos=0, retain=True)

    def publish_confidence(self, conf: float) -> None:
        """Optional: 0.0 .. 1.0 for proportional needle."""
        conf = max(0.0, min(1.0, float(conf)))
        self._client.publish(config.TOPIC_CONFIDENCE, f"{conf:.2f}", qos=0, retain=False)
