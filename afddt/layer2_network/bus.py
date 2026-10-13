"""Layer 2: Network layer.

Telemetry travels on topics shaped like MQTT topics:

    afddt/device/<id>/telemetry
    afddt/edge/<id>/telemetry
    afddt/gateway/<id>/telemetry

By default the bus runs inside the program, so nothing else needs installing.
If the MQTT_HOST setting points at a real broker (for example Eclipse
Mosquitto), messages are published to and received from that broker instead,
using the paho-mqtt client.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from typing import Callable

from ..config import SETTINGS


class MessageBus:
    def __init__(self):
        self._subs: dict[str, list[Callable]] = defaultdict(list)
        self.published = 0
        self.bytes = 0
        self.mode = "in-process"
        self._mqtt = None
        self._lock = threading.Lock()
        if SETTINGS.mqtt_host:
            self._connect_mqtt()

    def _connect_mqtt(self):
        try:
            import paho.mqtt.client as mqtt  # optional dependency
        except ImportError:
            print("[bus] MQTT_HOST is set but paho-mqtt is not installed; using the in-process bus")
            return
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) if hasattr(mqtt, "CallbackAPIVersion") else mqtt.Client()
            client.on_message = lambda c, u, msg: self._deliver(msg.topic, json.loads(msg.payload))
            client.connect(SETTINGS.mqtt_host, SETTINGS.mqtt_port, keepalive=30)
            client.subscribe("afddt/#")
            client.loop_start()
            self._mqtt = client
            self.mode = f"MQTT broker {SETTINGS.mqtt_host}:{SETTINGS.mqtt_port}"
        except Exception as exc:  # broker not reachable
            print(f"[bus] could not reach MQTT broker ({exc}); using the in-process bus")

    def subscribe(self, prefix: str, handler: Callable[[str, dict], None]) -> None:
        self._subs[prefix].append(handler)

    def publish(self, topic: str, payload: dict) -> None:
        data = json.dumps(payload, default=str)
        with self._lock:
            self.published += 1
            self.bytes += len(data)
        if self._mqtt is not None:
            self._mqtt.publish(topic, data)
        else:
            self._deliver(topic, payload)

    def _deliver(self, topic: str, payload: dict) -> None:
        for prefix, handlers in list(self._subs.items()):
            if topic.startswith(prefix):
                for h in handlers:
                    h(topic, payload)
