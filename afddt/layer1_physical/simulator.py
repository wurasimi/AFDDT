"""Layer 1: Physical layer.

There is no real hardware in this project, so this module simulates a small
IoT deployment: twelve end-devices of three classes, two edge/fog nodes and one
gateway.  Each tick it produces the telemetry those assets would send, and it
can inject attacks (scenarios) so the rest of the system has something to
detect.  Mitigations executed by the AFDDT are applied back to the simulated
assets, so a successful response visibly stops an attack.

Attacker addresses use the IP ranges reserved for documentation
(203.0.113.0/24 and 198.51.100.0/24), so no real address is ever involved.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import numpy as np

from ..config import SETTINGS

FW = {
    "smart_meter": "1.4.2",
    "env_sensor": "2.0.1",
    "ip_camera": "3.7.0",
}

# Benign behaviour of each device class: (mean, std) per signal
DEVICE_PROFILE = {
    "smart_meter": {"power_w": (3.0, 0.3), "pkt_rate": (2.0, 0.5), "pkt_size": (120, 15)},
    "env_sensor": {"power_w": (0.8, 0.1), "pkt_rate": (1.0, 0.3), "pkt_size": (80, 10)},
    "ip_camera": {"power_w": (5.5, 0.5), "pkt_rate": (40.0, 6.0), "pkt_size": (900, 80)},
}
CRITICAL_CLASSES = {"smart_meter"}


def fw_hash(device_class: str, version: str) -> str:
    return hashlib.sha256(f"{device_class}-firmware-{version}".encode()).hexdigest()


# Known-good firmware registry used by the Device Authentication & Firmware Integrity Agent
FIRMWARE_REGISTRY = {cls: fw_hash(cls, v) for cls, v in FW.items()}


@dataclass
class Asset:
    asset_id: str
    tier: str
    device_class: str = ""
    edge_id: str = ""
    ip: str = ""
    critical: bool = False
    isolated: bool = False
    rate_limited: bool = False
    firmware_version: str = ""
    firmware_hash: str = ""


@dataclass
class Scenario:
    """An attack being played against the simulated deployment."""
    name: str
    targets: list
    duration: int
    technique: list            # ground-truth ATT&CK technique(s), used only for scoring
    started: int = 0
    tick: int = 0
    params: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.tick < self.duration


SCENARIOS = {
    "brute_force": "Password guessing against a device from an external address",
    "firmware_tamper": "Unauthorised firmware written to a device",
    "resource_exhaustion": "Edge/fog node driven to CPU and memory exhaustion",
    "lateral_movement": "Compromised device reaches an edge node it is not attached to",
    "protocol_spoofing": "Spoofed frames injected through the gateway's protocol translation",
    "recon": "External port and service scanning through the gateway",
    "ddos_botnet": "Devices recruited into a botnet flood external targets",
    "mirai_campaign": "Multi-stage campaign: brute force, then lateral movement, then botnet flooding",
}


class IoTDeployment:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(SETTINGS.seed if seed is None else seed)
        self.pyrng = random.Random(SETTINGS.seed if seed is None else seed)
        self.tick = 0
        self.gateway = Asset("gw-01", "gateway", ip="10.0.0.1")
        self.edges: dict[str, Asset] = {}
        self.devices: dict[str, Asset] = {}
        for e in range(1, SETTINGS.edge_nodes + 1):
            eid = f"edge-{e:02d}"
            self.edges[eid] = Asset(eid, "edge", ip=f"10.0.{e}.1")
        n = 0
        for cls in SETTINGS.device_classes:
            for i in range(1, SETTINGS.devices_per_class + 1):
                edge_id = list(self.edges)[n % len(self.edges)]
                did = f"{cls.replace('_', '-')}-{i:02d}"
                e_no = int(edge_id[-2:])
                self.devices[did] = Asset(
                    did, "device", device_class=cls, edge_id=edge_id,
                    ip=f"10.0.{e_no}.{10 + n}", critical=cls in CRITICAL_CLASSES,
                    firmware_version=FW[cls], firmware_hash=FIRMWARE_REGISTRY[cls],
                )
                n += 1
        self.gateway_segmented = False
        self.blocked_ips: set[str] = set()
        self.scenarios: list[Scenario] = []

    # ------------------------------------------------------------------ topology
    def topology(self) -> dict:
        return {
            "gateway": {"id": self.gateway.asset_id, "ip": self.gateway.ip},
            "edges": [{"id": e.asset_id, "ip": e.ip} for e in self.edges.values()],
            "devices": [{"id": d.asset_id, "class": d.device_class, "edge": d.edge_id, "ip": d.ip,
                         "critical": d.critical, "firmware_version": d.firmware_version}
                        for d in self.devices.values()],
        }

    # ------------------------------------------------------------------ attacks
    def start_scenario(self, name: str, target: str | None = None) -> Scenario:
        devs = [d for d in self.devices.values() if not d.isolated]
        pick = lambda cls=None: (self.devices[target] if target in self.devices else
                                 self.pyrng.choice([d for d in devs if cls is None or d.device_class == cls] or devs))
        if name == "brute_force":
            d = pick()
            sc = Scenario(name, [d.asset_id], 25, ["T1110"], params={"ip": "203.0.113.45"})
        elif name == "firmware_tamper":
            d = pick()
            sc = Scenario(name, [d.asset_id], 25, ["T1601"])
        elif name == "resource_exhaustion":
            e = self.edges[target] if target in self.edges else self.pyrng.choice(list(self.edges.values()))
            sc = Scenario(name, [e.asset_id], 25, ["T1499"])
        elif name == "lateral_movement":
            d = pick()
            other = [e for e in self.edges if e != d.edge_id][0]
            sc = Scenario(name, [d.asset_id, other], 25, ["T1021"])
        elif name == "protocol_spoofing":
            sc = Scenario(name, [self.gateway.asset_id], 25, ["T1557"], params={"ip": "198.51.100.23"})
        elif name == "recon":
            sc = Scenario(name, [self.gateway.asset_id], 25, ["T1595", "T1046"], params={"ip": "203.0.113.77"})
        elif name == "ddos_botnet":
            bots = self.pyrng.sample(devs, k=min(3, len(devs)))
            sc = Scenario(name, [b.asset_id for b in bots] + [self.gateway.asset_id], 30, ["T1498"],
                          params={"c2": "203.0.113.66"})
        elif name == "mirai_campaign":
            d = pick("ip_camera")
            other = [e for e in self.edges if e != d.edge_id][0]
            sc = Scenario(name, [d.asset_id, other, self.gateway.asset_id], 45, ["T1110", "T1021", "T1498"],
                          params={"ip": "203.0.113.45", "c2": "203.0.113.66"})
        else:
            raise ValueError(f"unknown scenario {name}")
        sc.started = self.tick
        self.scenarios.append(sc)
        return sc

    def active_ground_truth(self) -> dict:
        """twin_id -> set of techniques currently being played against it (for live scoring only)."""
        gt: dict[str, set] = {}
        for sc in self.scenarios:
            if sc.active or self.tick - (sc.started + sc.duration) < 15:
                for t in sc.targets:
                    gt.setdefault(t, set()).update(sc.technique)
        return gt

    # ------------------------------------------------------------------ mitigations
    def apply_mitigation(self, action_type: str, target: str, detail: dict | None = None) -> str:
        detail = detail or {}
        if action_type == "isolate_device" and target in self.devices:
            self.devices[target].isolated = True
            return f"{target} isolated from the network"
        if action_type == "rate_limit_device" and target in self.devices:
            self.devices[target].rate_limited = True
            return f"traffic from {target} rate-limited"
        if action_type in ("reflash_firmware",) and target in self.devices:
            d = self.devices[target]
            d.isolated = True
            d.firmware_version = FW[d.device_class]
            d.firmware_hash = FIRMWARE_REGISTRY[d.device_class]
            for sc in self.scenarios:
                if sc.name == "firmware_tamper" and target in sc.targets:
                    sc.tick = sc.duration
            d.isolated = False
            return f"{target} re-flashed with known-good firmware {d.firmware_version}"
        if action_type == "block_source_ip":
            ip = detail.get("ip") or target
            self.blocked_ips.add(ip)
            return f"inbound traffic from {ip} blocked at the gateway"
        if action_type == "segment_network":
            self.gateway_segmented = True
            return "outbound traffic from the IoT segment restricted at the gateway"
        if action_type == "throttle_edge" and target in self.edges:
            for sc in self.scenarios:
                if sc.name == "resource_exhaustion" and target in sc.targets:
                    sc.tick = sc.duration
            return f"rogue jobs on {target} stopped and workload throttled"
        if action_type == "quarantine_link":
            dev = detail.get("device")
            if dev in self.devices:
                self.devices[dev].isolated = True
            return f"link from {dev} to {target} quarantined"
        if action_type == "reset_gateway_bindings":
            for sc in self.scenarios:
                if sc.name == "protocol_spoofing":
                    sc.tick = sc.duration
            return "gateway ARP/address bindings reset and source validation enforced"
        if action_type in ("monitor", "notify"):
            return f"{target} placed under heightened monitoring"
        return "no change"

    def restore(self) -> None:
        for d in self.devices.values():
            d.isolated = False
            d.rate_limited = False
            d.firmware_version = FW[d.device_class]
            d.firmware_hash = FIRMWARE_REGISTRY[d.device_class]
        self.gateway_segmented = False
        self.blocked_ips.clear()
        self.scenarios.clear()

    # ------------------------------------------------------------------ telemetry
    def _n(self, mean, std):
        return float(max(0.0, self.rng.normal(mean, std)))

    def step(self) -> dict:
        """Advance one tick and return the telemetry of every asset."""
        self.tick += 1
        r = self.rng
        dev_msgs, edge_msgs = {}, {}

        for d in self.devices.values():
            p = DEVICE_PROFILE[d.device_class]
            m = {
                "tick": self.tick, "twin_id": d.asset_id, "device_class": d.device_class, "ip": d.ip,
                "auth_failures": int(r.poisson(0.05)), "auth_success": int(r.poisson(0.2)),
                "power_w": self._n(*p["power_w"]), "pkt_rate": self._n(*p["pkt_rate"]),
                "pkt_size": self._n(*p["pkt_size"]), "malformed": int(r.poisson(0.02)),
                "unique_dst": int(1 + r.poisson(0.3)),
                "firmware_version": d.firmware_version, "firmware_hash": d.firmware_hash,
                "remote_ips": [], "isolated": d.isolated,
            }
            dev_msgs[d.asset_id] = m

        for e in self.edges.values():
            attached = [d.asset_id for d in self.devices.values() if d.edge_id == e.asset_id]
            load = sum(dev_msgs[a]["pkt_rate"] for a in attached if not self.devices[a].isolated)
            edge_msgs[e.asset_id] = {
                "tick": self.tick, "twin_id": e.asset_id,
                "cpu": min(100.0, self._n(28 + 0.08 * min(load, 150), 4)), "mem": self._n(45, 3),
                "battery": self._n(82, 1.5), "jobs": int(max(1, r.normal(5, 1))),
                "connections": [a for a in attached if not self.devices[a].isolated],
            }

        gw = {
            "tick": self.tick, "twin_id": self.gateway.asset_id,
            "translations": 0.0, "translation_errors": int(r.poisson(0.3)),
            "spoofed_frames": 0, "malformed_frames": int(r.poisson(0.2)),
            "out_pkts": 0.0, "unique_dst": int(max(1, r.normal(6, 1))),
            "syn_ratio": float(np.clip(r.normal(0.06, 0.02), 0, 1)),
            "udp_ratio": float(np.clip(r.normal(0.40, 0.05), 0, 1)),
            "icmp_ratio": float(np.clip(r.normal(0.02, 0.01), 0, 1)),
            "inbound_unique_ports": int(max(0, r.normal(3, 1))),
            "new_routes": 0, "external_ips": [], "segmented": self.gateway_segmented,
        }

        # ----------------------------------------------- apply active attacks
        for sc in self.scenarios:
            if not sc.active:
                continue
            k = sc.tick
            if sc.name == "brute_force":
                m = dev_msgs[sc.targets[0]]
                if not self.devices[sc.targets[0]].isolated and sc.params["ip"] not in self.blocked_ips:
                    m["auth_failures"] = int(r.integers(15, 40))
                    m["remote_ips"] = [sc.params["ip"]]
                    gw["external_ips"].append(sc.params["ip"])
            elif sc.name == "firmware_tamper":
                d = self.devices[sc.targets[0]]
                d.firmware_version = FW[d.device_class] + "-x"
                d.firmware_hash = fw_hash(d.device_class, "tampered")
                m = dev_msgs[d.asset_id]
                m["firmware_version"], m["firmware_hash"] = d.firmware_version, d.firmware_hash
                m["power_w"] *= 1.15
            elif sc.name == "resource_exhaustion":
                e = edge_msgs[sc.targets[0]]
                e["cpu"] = min(100.0, self._n(96, 2))
                e["mem"] = min(100.0, self._n(93, 2))
                e["jobs"] = int(r.integers(35, 60))
                e["battery"] = max(0.0, e["battery"] - 2.5 * k)
            elif sc.name == "lateral_movement":
                dev, other = sc.targets
                if not self.devices[dev].isolated:
                    edge_msgs[other]["connections"].append(dev)
                    edge_msgs[other]["jobs"] += 2
                    dev_msgs[dev]["unique_dst"] += 4
            elif sc.name == "protocol_spoofing":
                if sc.params["ip"] not in self.blocked_ips:
                    gw["translation_errors"] = int(r.integers(20, 45))
                    gw["spoofed_frames"] = int(r.integers(15, 35))
                    gw["new_routes"] = int(r.integers(2, 6))
                    gw["external_ips"].append(sc.params["ip"])
            elif sc.name == "recon":
                if sc.params["ip"] not in self.blocked_ips:
                    gw["inbound_unique_ports"] = int(r.integers(150, 400))
                    gw["syn_ratio"] = float(np.clip(r.normal(0.65, 0.05), 0, 1))
                    gw["unique_dst"] = int(r.integers(10, 15))
                    gw["external_ips"].append(sc.params["ip"])
            elif sc.name == "ddos_botnet":
                self._botnet(sc.targets[:-1], dev_msgs, gw, sc.params["c2"])
            elif sc.name == "mirai_campaign":
                dev, other, _ = sc.targets
                if k < 12:      # stage 1: low-and-slow credential guessing
                    if not self.devices[dev].isolated and sc.params["ip"] not in self.blocked_ips:
                        dev_msgs[dev]["auth_failures"] = int(r.integers(2, 5))
                        dev_msgs[dev]["remote_ips"] = [sc.params["ip"]]
                        gw["external_ips"].append(sc.params["ip"])
                elif k < 22:    # stage 2: lateral movement
                    if not self.devices[dev].isolated:
                        edge_msgs[other]["connections"].append(dev)
                        edge_msgs[other]["jobs"] += 2
                        dev_msgs[dev]["unique_dst"] += 4
                        dev_msgs[dev]["pkt_rate"] *= 2.5
                else:           # stage 3: flooding
                    self._botnet([dev], dev_msgs, gw, sc.params["c2"])
            sc.tick += 1

        # ----------------------------------------------- mitigation effects
        for did, m in dev_msgs.items():
            d = self.devices[did]
            if d.isolated:
                m["pkt_rate"], m["unique_dst"], m["auth_failures"], m["remote_ips"] = 0.0, 0, 0, []
            elif d.rate_limited:
                m["pkt_rate"] = min(m["pkt_rate"], DEVICE_PROFILE[d.device_class]["pkt_rate"][0] * 1.5)
        live = [m for did, m in dev_msgs.items() if not self.devices[did].isolated]
        gw["translations"] = float(sum(m["pkt_rate"] for m in live) * 0.5 + r.normal(0, 1))
        base_out = sum(min(m["pkt_rate"], 60) for m in live)
        gw["out_pkts"] = max(gw["out_pkts"], base_out)
        if self.gateway_segmented:
            gw["out_pkts"] = min(gw["out_pkts"], base_out)
            gw["unique_dst"] = min(gw["unique_dst"], 8)
            gw["udp_ratio"] = min(gw["udp_ratio"], 0.5)
        gw["out_pkts_per_device"] = gw["out_pkts"] / max(1, len(live))
        gw["routing_table"] = {d.ip: d.edge_id for d in self.devices.values() if not d.isolated}
        gw["external_ips"] = sorted(set(gw["external_ips"]) - self.blocked_ips)
        return {"tick": self.tick, "devices": dev_msgs, "edges": edge_msgs, "gateway": gw}

    def _botnet(self, bots, dev_msgs, gw, c2):
        r = self.rng
        active = [b for b in bots if not self.devices[b].isolated]
        for b in active:
            dev_msgs[b]["pkt_rate"] = float(r.normal(900, 80))
            dev_msgs[b]["pkt_size"] = float(r.normal(64, 6))
            dev_msgs[b]["unique_dst"] = int(r.integers(30, 60))
            dev_msgs[b]["remote_ips"] = [c2]
        if active and not self.gateway_segmented:
            gw["out_pkts"] = float(r.normal(900, 80)) * len(active)
            gw["unique_dst"] = int(r.integers(40, 90))
            gw["udp_ratio"] = float(np.clip(r.normal(0.9, 0.03), 0, 1))
            gw["new_routes"] = int(r.integers(1, 4))
            gw["external_ips"].append(c2)

    def benign_samples(self, n: int) -> list[dict]:
        """Benign telemetry used to train the Layer 4 models before the system starts."""
        saved = (self.tick, list(self.scenarios))
        self.scenarios = []
        out = [self.step() for _ in range(n)]
        self.tick, self.scenarios = saved
        return out
