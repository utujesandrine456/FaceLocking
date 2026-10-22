"""
pc/timer_pub.py — test MQTT without camera.
Publishes face/status 1 then 0 every few seconds.
"""
from __future__ import annotations

import time

from mqtt_client import FaceMqttClient


def main():
    mqtt = FaceMqttClient()
    mqtt.connect()
    print("Timer publisher: toggles face/status every 4s. Ctrl+C to stop.")
    try:
        while True:
            mqtt.publish_status("1", force=True)
            mqtt.publish_name("TEST")
            time.sleep(4)
            mqtt.publish_status("0", force=True)
            mqtt.publish_name("")
            time.sleep(4)
    except KeyboardInterrupt:
        pass
    finally:
        mqtt.close()


if __name__ == "__main__":
    main()
