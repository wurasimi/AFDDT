"""The digital twins (Section 3.5 of the dissertation).

Each physical asset has one twin, implemented as a *shadow object*: a record of
its identity and recent state, kept in sync with the asset.  Twins hold state
only.  Agency is supplied separately by shared agents in layer7_agents, which
read many twins through the service layer.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

WINDOW = 60  # ticks of telemetry kept in each shadow object


@dataclass
class DeviceTwin:
    twin_id: str
    device_class: str
    edge_id: str
    gateway_id: str
    ip: str
    critical: bool
    firmware_version: str = ""
    firmware_hash: str = ""
    status: str = "normal"            # normal | suspect | isolated | rate_limited
    telemetry: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    auth_log: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    last_sync: float = 0.0

    def sync(self, msg: dict) -> None:
        self.telemetry.append(msg)
        self.auth_log.append({"tick": msg["tick"], "failures": msg["auth_failures"],
                              "success": msg["auth_success"], "remote_ips": msg.get("remote_ips", [])})
        self.firmware_version = msg["firmware_version"]
        self.firmware_hash = msg["firmware_hash"]
        self.last_sync = time.time()


@dataclass
class EdgeFogTwin:
    twin_id: str
    gateway_id: str
    attached_device_ids: list
    status: str = "normal"
    telemetry: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    last_sync: float = 0.0

    def sync(self, msg: dict) -> None:
        self.telemetry.append(msg)
        self.last_sync = time.time()


@dataclass
class GatewayTwin:
    twin_id: str
    status: str = "normal"            # normal | segmented
    routing_table: dict = field(default_factory=dict)
    protocol_log: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    traffic: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    last_sync: float = 0.0

    def sync(self, msg: dict) -> None:
        self.protocol_log.append({k: msg[k] for k in ("tick", "translations", "translation_errors",
                                                      "spoofed_frames", "malformed_frames")})
        self.traffic.append(msg)
        self.routing_table = msg.get("routing_table", self.routing_table)
        self.last_sync = time.time()


@dataclass
class AFDDT:
    """The fused domain twin: aggregates every device, edge and gateway twin."""
    devices: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)
    gateways: dict = field(default_factory=dict)

    def get(self, twin_id: str):
        return self.devices.get(twin_id) or self.edges.get(twin_id) or self.gateways.get(twin_id)

    def tier_of(self, twin_id: str) -> str:
        if twin_id in self.devices:
            return "device"
        if twin_id in self.edges:
            return "edge"
        return "gateway"
