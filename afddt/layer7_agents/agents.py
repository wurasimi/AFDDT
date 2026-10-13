"""Layer 7.1: the twelve AI agents (Table 3.7 of the dissertation).

Each agent is ONE shared instance that reasons over every twin of its type
(Section 3.4.3): agency is delivered per device because each device gets its
own finding, not because each device gets its own agent.

    Monitor   1 DeviceAuthFirmwareAgent      2 DeviceBehaviourAgent
              3 ResourceExhaustionAgent      4 LateralMovementAgent
              5 ProtocolTranslationAgent     6 TrafficBotnetAgent
    Analyze   7 CrossDomainFusionAgent       8 ThreatEnrichmentAgent
    Plan      9 ResponsePlanningAgent       10 MitigationSimulationAgent
    Execute  11 ActionExecutionAgent
    Learn    12 LearningFeedbackAgent
(The Orchestrator, agent 13, is in orchestrator.py.)
"""
from __future__ import annotations

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..layer1_physical.simulator import FIRMWARE_REGISTRY
from ..layer4_models.detectors import DEVICE_FEATURES, EDGE_FEATURES, GW_TRANSLATION_FEATURES, GW_VOLUME_FEATURES
from ..layer5_xai.explain import attribution, attribution_text, enrichment_text, observations

SEVERITY = ["low", "medium", "high", "critical"]
_ids = itertools.count(1)


@dataclass
class Finding:
    tier: str
    source_agent: str
    twin_ids: list
    confidence: float
    summary: str
    observations: list
    explanation: dict
    tick: int
    t0: float                         # perf_counter when the triggering telemetry arrived
    remote_ips: list = field(default_factory=list)
    finding_id: str = ""
    ts: float = field(default_factory=time.time)
    severity: str = "low"
    base_severity: str = "low"
    status: str = "new"               # new | held | fused | processed
    fused_from: list = field(default_factory=list)
    enrichment_status: str = "pending"
    enrichment: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)
    decision: str = ""
    escalated: bool = False
    latency_ms: float = 0.0
    ground_truth: list = field(default_factory=list)
    feedback: str = ""
    members: list = field(default_factory=list)   # constituent findings of a fused finding
    hold_until: int = 0                             # findings wait briefly so fusion can correlate them
    wait_ticks: int = 0                             # ticks spent waiting for correlation before processing
    process_ms: float = 0.0                         # compute time of the MAPE-K loop for this finding

    def __post_init__(self):
        if not self.finding_id:
            self.finding_id = f"F-{next(_ids):04d}"


@dataclass
class Action:
    finding_id: str
    type: str
    target: str
    risk: str
    description: str
    detail: dict = field(default_factory=dict)
    action_id: str = ""
    ts: float = field(default_factory=time.time)
    simulation: dict = field(default_factory=dict)
    status: str = "proposed"          # proposed | executed | awaiting_approval | rejected | dismissed
    approved_by: str = ""
    outcome: str = ""

    def __post_init__(self):
        if not self.action_id:
            self.action_id = f"A-{next(_ids):04d}"


class Agent:
    number = 0
    name = ""
    stage = ""
    twin = "all"

    def __init__(self, svc, log):
        self.svc = svc
        self.log = log                  # callback(agent_name, event, detail)
        self.count = 0
        self.last = "idle"

    def note(self, msg: str) -> None:
        self.count += 1
        self.last = msg

    def info(self) -> dict:
        return {"no": self.number, "name": self.name, "stage": self.stage, "twin": self.twin,
                "count": self.count, "last": self.last}


def base_severity(conf: float) -> int:
    """Severity from detection confidence alone, before any threat intelligence is applied."""
    return 1 if conf >= 0.7 else 0


# ====================================================================== MONITOR
class MonitorAgent(Agent):
    stage = "Monitor"

    def __init__(self, svc, log, cooldown: int):
        super().__init__(svc, log)
        self.cooldown = cooldown
        self.last_report: dict[str, int] = {}

    def cooling(self, key: str, tick: int) -> bool:
        return tick - self.last_report.get(key, -10_000) < self.cooldown

    def report(self, key: str, f: Finding) -> Finding:
        self.last_report[key] = f.tick
        f.base_severity = SEVERITY[base_severity(f.confidence)]
        f.severity = f.base_severity
        self.note(f"{f.finding_id}: {f.summary}")
        self.log(self.name, "finding", {"finding": f.finding_id, "summary": f.summary})
        return f


class DeviceAuthFirmwareAgent(MonitorAgent):
    number, name, twin = 1, "Device Authentication & Firmware Integrity", "device"

    def __init__(self, svc, log, cooldown):
        super().__init__(svc, log, cooldown)
        self.reported_hash: dict[str, str] = {}

    def run(self, tick: int, t0: float) -> list[Finding]:
        out = []
        for did, dev in self.svc.afddt.devices.items():
            if dev.status == "isolated" or not dev.telemetry:
                continue
            msg = dev.telemetry[-1]
            bad_fw = msg["firmware_hash"] != FIRMWARE_REGISTRY[dev.device_class]
            if bad_fw and self.reported_hash.get(did) != msg["firmware_hash"]:
                self.reported_hash[did] = msg["firmware_hash"]
                out.append(self.report(did + "fw", Finding(
                    "device", self.name, [did], 0.95,
                    f"Firmware on {did} (v{msg['firmware_version']}) does not match the known-good image",
                    ["firmware image hash does not match the known-good firmware for this embedded device",
                     "modified firmware version"],
                    {"type": "rule", "text": f"hash {msg['firmware_hash'][:12]}... not in firmware registry"},
                    tick, t0)))
            recent = list(dev.auth_log)[-3:]
            fails = sum(a["failures"] for a in recent)
            slow = list(dev.auth_log)[-15:]
            slow_fails = sum(a["failures"] for a in slow)
            if fails < 15 and slow_fails >= 25 and not self.cooling(did + "auth", tick):
                ips = sorted({ip for a in slow for ip in a["remote_ips"]})
                out.append(self.report(did + "auth", Finding(
                    "device", self.name, [did], 0.62,
                    f"Low-and-slow password guessing on {did}: {slow_fails} failed logins in 15 ticks"
                    + (f" from {', '.join(ips)}" if ips else ""),
                    ["many repeated failed authentication login attempts guessing passwords", "slow persistent attempts"],
                    {"type": "rule", "text": f"{slow_fails} failures in the last 15 ticks (threshold 25)"},
                    tick, t0, remote_ips=ips)))
                continue
            if fails >= 15 and not self.cooling(did + "auth", tick):
                model = self.svc.ml.device_model(dev.device_class)
                x = np.array([float(msg.get(k, 0)) for k in DEVICE_FEATURES])
                attr = attribution(DEVICE_FEATURES, x, model.mean, model.std)
                ips = sorted({ip for a in recent for ip in a["remote_ips"]})
                conf = float(min(0.97, 0.55 + fails / 100))
                out.append(self.report(did + "auth", Finding(
                    "device", self.name, [did], conf,
                    f"{fails} failed logins on {did} in 3 ticks" + (f" from {', '.join(ips)}" if ips else ""),
                    observations(attr) or ["many repeated failed authentication login attempts"],
                    {"type": "feature_attribution", "attribution": attr, "text": attribution_text(attr)},
                    tick, t0, remote_ips=ips)))
        return out


class DeviceBehaviourAgent(MonitorAgent):
    number, name, twin = 2, "Device Behaviour Anomaly", "device"

    def __init__(self, svc, log, cooldown):
        super().__init__(svc, log, cooldown)
        self.streak: dict[str, int] = defaultdict(int)

    def run(self, tick: int, t0: float) -> list[Finding]:
        out = []
        by_class: dict[str, list] = defaultdict(list)
        for did, dev in self.svc.afddt.devices.items():
            if dev.status != "isolated" and dev.telemetry:
                by_class[dev.device_class].append(dev)
        for cls, devs in by_class.items():
            scores, X = self.svc.ml.score_devices(cls, [d.telemetry[-1] for d in devs])
            model = self.svc.ml.device_model(cls)
            for dev, score, x in zip(devs, scores, X):
                did, msg = dev.twin_id, dev.telemetry[-1]
                attr = attribution(DEVICE_FEATURES, x, model.mean, model.std) if score > 0 else []
                # authentication spikes belong to agent 1; this agent reports other behaviour
                attr = [a for a in attr if a["feature"] != "auth_failures"]
                self.streak[did] = self.streak[did] + 1 if (score > 0 and attr) else 0
                if self.streak[did] >= 2 and not self.cooling(did, tick):
                    conf = float(min(0.95, 0.6 + score))
                    out.append(self.report(did, Finding(
                        "device", self.name, [did], conf,
                        f"{did} behaving abnormally: {', '.join(a['feature'] for a in attr)}",
                        observations(attr),
                        {"type": "feature_attribution", "attribution": attr, "text": attribution_text(attr),
                         "anomaly_score": round(float(score), 3)},
                        tick, t0, remote_ips=list(msg.get("remote_ips", [])))))
        return out


class ResourceExhaustionAgent(MonitorAgent):
    number, name, twin = 3, "Resource Exhaustion Monitoring", "edge"

    def __init__(self, svc, log, cooldown):
        super().__init__(svc, log, cooldown)
        self.streak: dict[str, int] = defaultdict(int)

    def run(self, tick: int, t0: float) -> list[Finding]:
        out = []
        models = self.svc.models
        for eid, edge in self.svc.afddt.edges.items():
            if not edge.telemetry:
                continue
            msg = edge.telemetry[-1]
            z = self.svc.ml.edge_z(eid, msg)
            hot = max(z[0], z[1], z[2]) > 4 or z[3] < -4
            self.streak[eid] = self.streak[eid] + 1 if hot else 0
            if self.streak[eid] >= 3 and not self.cooling(eid, tick):
                x = np.array([float(msg[k]) for k in EDGE_FEATURES])
                attr = attribution(EDGE_FEATURES, x, models.edge_mean[eid], models.edge_std[eid])
                out.append(self.report(eid, Finding(
                    "edge", self.name, [eid], 0.9,
                    f"{eid} resources exhausted: CPU {msg['cpu']:.0f}%, memory {msg['mem']:.0f}%, {msg['jobs']} jobs",
                    observations(attr),
                    {"type": "feature_attribution", "attribution": attr, "text": attribution_text(attr)},
                    tick, t0)))
        return out


class LateralMovementAgent(MonitorAgent):
    number, name, twin = 4, "Lateral Movement Detection", "edge"

    def run(self, tick: int, t0: float) -> list[Finding]:
        out = []
        kg = self.svc.data.graph()
        for eid, edge in self.svc.afddt.edges.items():
            if not edge.telemetry:
                continue
            for dev in edge.telemetry[-1]["connections"]:
                if kg.observe_connection(dev, eid, tick) and not self.cooling(dev + eid, tick):
                    home = kg.attached_edge(dev)
                    path = [dev, home, "gw-01", eid]
                    out.append(self.report(dev + eid, Finding(
                        "edge", self.name, [dev, eid], 0.85,
                        f"{dev} (attached to {home}) opened a connection to {eid}",
                        ["device opened connections to a different node outside its normal attachment",
                         "moving laterally between systems in the network"],
                        {"type": "graph_path", "path": path,
                         "text": f"new edge {dev} -> {eid} is not in the topology; normal path is {dev} -> {home}"},
                        tick, t0)))
        return out


class ProtocolTranslationAgent(MonitorAgent):
    number, name, twin = 5, "Protocol Translation Security", "gateway"

    def __init__(self, svc, log, cooldown):
        super().__init__(svc, log, cooldown)
        self.streak = 0

    def run(self, tick: int, t0: float) -> list[Finding]:
        gw = next(iter(self.svc.afddt.gateways.values()))
        if not gw.traffic:
            return []
        msg = gw.traffic[-1]
        score, x = self.svc.ml.score_gateway_translation(msg)
        self.streak = self.streak + 1 if score > 0 else 0
        if self.streak >= 2 and not self.cooling(gw.twin_id, tick):
            m = self.svc.models.gw_translation
            attr = attribution(GW_TRANSLATION_FEATURES, x, m.mean, m.std)
            return [self.report(gw.twin_id, Finding(
                "gateway", self.name, [gw.twin_id], float(min(0.95, 0.65 + score)),
                f"Abnormal protocol translation at {gw.twin_id}: {msg['translation_errors']} errors, "
                f"{msg['spoofed_frames']} spoofed frames",
                observations(attr),
                {"type": "feature_attribution", "attribution": attr, "text": attribution_text(attr)},
                tick, t0, remote_ips=list(msg.get("external_ips", []))))]
        return []


class TrafficBotnetAgent(MonitorAgent):
    """Gateway findings are HELD for fusion: a gateway alone cannot always tell
    legitimate growth from botnet staging (Section 3.5.3)."""
    number, name, twin = 6, "Traffic & Botnet Detection", "gateway"

    def __init__(self, svc, log, cooldown):
        super().__init__(svc, log, cooldown)
        self.streak = 0

    def run(self, tick: int, t0: float) -> list[Finding]:
        gw = next(iter(self.svc.afddt.gateways.values()))
        if not gw.traffic:
            return []
        msg = gw.traffic[-1]
        score, x = self.svc.ml.score_gateway_volume(msg)
        self.streak = self.streak + 1 if score > 0 else 0
        if self.streak >= 2 and not self.cooling(gw.twin_id, tick):
            m = self.svc.models.gw_volume
            attr = attribution(GW_VOLUME_FEATURES, x, m.mean, m.std)
            f = self.report(gw.twin_id, Finding(
                "gateway", self.name, [gw.twin_id], float(min(0.9, 0.6 + score)),
                f"Traffic anomaly at {gw.twin_id}: {msg['out_pkts']:.0f} pkts out, {msg['unique_dst']} destinations, "
                f"{msg['inbound_unique_ports']} probed ports",
                observations(attr),
                {"type": "flow_evidence", "attribution": attr, "text": attribution_text(attr)},
                tick, t0, remote_ips=list(msg.get("external_ips", []))))
            f.status = "held"
            f.hold_until = tick + self.svc.gateway_hold
            return [f]
        return []


# ====================================================================== ANALYZE
class CrossDomainFusionAgent(Agent):
    number, name, stage = 7, "Cross-Domain Threat Fusion", "Analyze"

    def __init__(self, svc, log, window: int, hold: int):
        super().__init__(svc, log)
        self.window, self.hold = window, hold
        self.open_fused: list[Finding] = []

    @staticmethod
    def _linked(a: Finding, b: Finding, kg) -> str | None:
        if a.tier == b.tier and a.tier != "cross-domain":
            return None
        shared_ip = set(a.remote_ips) & set(b.remote_ips)
        if shared_ip:
            return f"shared external address {', '.join(sorted(shared_ip))}"
        shared_twin = set(a.twin_ids) & set(b.twin_ids)
        if shared_twin:
            return f"same twin {', '.join(sorted(shared_twin))}"
        if any(f.tier == "gateway" and f.source_agent.startswith("Traffic") for f in (a, b)):
            vol = [f for f in (a, b) if f.tier == "device" and any(
                "packet rate" in o or "destinations" in o or "flood" in o for o in f.observations)]
            if vol:
                return f"device outbound surge on {vol[0].twin_ids[0]} coincides with gateway traffic anomaly"
        return None

    def run(self, tick: int, recent: list[Finding]) -> tuple[list[Finding], list[Finding]]:
        """Returns (new or updated fused findings, held findings released as standalone)."""
        kg = self.svc.data.graph()
        window = [f for f in recent if tick - f.tick <= self.window and f.tier != "cross-domain"]
        self.open_fused = [f for f in self.open_fused if tick - f.tick <= self.window]
        produced, released = [], []
        for f in [f for f in window if f.status in ("new", "held")]:
            partners = [(g, why) for g in window if g is not f
                        for why in [self._linked(f, g, kg)] if why]
            if not partners:
                continue
            target = next((x for x in self.open_fused
                           if set(x.fused_from) & ({f.finding_id} | {g.finding_id for g, _ in partners})), None)
            members = [f] + [g for g, _ in partners]
            reasons = [why for _, why in partners]
            if target is None:
                target = Finding("cross-domain", self.name, [], 0.0, "", [], {"type": "fusion", "links": []},
                                 tick, f.t0)
                self.open_fused.append(target)
            tiers_before = {self.svc.afddt.tier_of(t) for t in target.twin_ids}
            for m in members:
                if m.finding_id not in target.fused_from:
                    target.fused_from.append(m.finding_id)
                    target.twin_ids = sorted(set(target.twin_ids) | set(m.twin_ids))
                    target.observations = list(dict.fromkeys(target.observations + m.observations))
                    target.remote_ips = sorted(set(target.remote_ips) | set(m.remote_ips))
                if m not in target.members:
                    target.members.append(m)
                if m.status in ("new", "held"):
                    m.status = "fused"
            target.explanation["links"] = list(dict.fromkeys(target.explanation["links"] + reasons))
            confs = [m.confidence for m in members]
            target.confidence = round(float(1 - np.prod([1 - c for c in confs])), 3)
            tiers = sorted({self.svc.afddt.tier_of(t) for t in target.twin_ids})
            target.summary = (f"Multi-stage activity across {', '.join(tiers)} tiers involving "
                              f"{', '.join(target.twin_ids[:5])}")
            target.explanation["text"] = "; ".join(target.explanation["links"])
            target.t0 = min(target.t0, f.t0)
            if set(tiers) != tiers_before or target.status == "new":
                target.base_severity = SEVERITY[base_severity(target.confidence)]
                target.severity = target.base_severity
                target.status = "new"
                if target not in produced:
                    produced.append(target)
        for f in window:
            if f.status == "held" and tick >= f.hold_until:
                f.status = "new"
                if f.source_agent.startswith("Traffic"):
                    f.confidence = round(f.confidence * 0.85, 3)
                    f.summary += " (no corroborating device evidence)"
                released.append(f)
        for f in produced:
            self.note(f"{f.finding_id}: {f.summary}")
            self.log(self.name, "fused", {"finding": f.finding_id, "from": f.fused_from, "links": f.explanation["links"]})
        return produced, released


TACTIC_WEIGHT = {"impact": 2, "lateral-movement": 1, "initial-access": 1, "credential-access": 1,
                 "defense-evasion": 1, "command-and-control": 1, "collection": 1, "persistence": 1,
                 "discovery": 0, "reconnaissance": 0, "resource-development": 0}


def severity_after_enrichment(base: str, tactics: list, ioc: bool, cross_domain: bool) -> str:
    """Severity grounded in threat intelligence (Section 3.6.4)."""
    level = SEVERITY.index(base)
    if "impact" in tactics:
        level += 1
    if ioc and ("impact" in tactics or "command-and-control" in tactics):
        level += 1
    if cross_domain:
        level += 1
    return SEVERITY[min(3, level)]


class ThreatEnrichmentAgent(Agent):
    number, name, stage = 8, "Threat Enrichment", "Analyze"

    def _match(self, observations: list[str], remote_ips: list[str]) -> dict | None:
        query = ". ".join(observations)
        cands = self.svc.data.enrichment_lookup(query, k=5)
        best: dict[str, dict] = {}
        for c in cands:      # learned examples collapse onto their technique
            if c["ext_id"] not in best or c["similarity"] > best[c["ext_id"]]["similarity"]:
                best[c["ext_id"]] = dict(c, kind="technique")
        kg = self.svc.data.graph()
        iocs = self.svc.data.indicators_for(remote_ips)
        ioc_techs, campaign = set(), None
        for ip, node in iocs:
            for t in kg.related(node, "technique"):
                ioc_techs.add(kg.g.nodes[t]["ext_id"])
            for c in kg.related(node, "campaign"):
                campaign = kg.g.nodes[c]["label"]
        for tid in ioc_techs:
            if tid in best:
                best[tid]["similarity"] = round(best[tid]["similarity"] + 0.15, 3)
        ranked = sorted(best.values(), key=lambda c: -c["similarity"])[:5]
        if not ranked or ranked[0]["similarity"] < 0.05:
            return None
        chosen, engine = self.svc.cognitive.choose_technique(observations, ranked[:3])
        top = next((c for c in ranked if c["ext_id"] == chosen), ranked[0])
        tech = self.svc.corpus.techniques().get(top["ext_id"])
        return {"top": top, "ranked": ranked, "engine": engine, "iocs": [ip for ip, _ in iocs],
                "campaign": campaign, "tactics": tech.tactics if tech else top.get("tactics", [])}

    def enrich(self, f: Finding) -> Finding:
        kg = self.svc.data.graph()
        if f.members:
            # fused finding: classify each stage, then lead with the most damaging technique
            stages = []
            for m in sorted(f.members, key=lambda m: m.tick):
                r = self._match(m.observations, m.remote_ips)
                if r:
                    stages.append(r)
            chain = list(dict.fromkeys(s["top"]["ext_id"] for s in stages))
            if stages:
                r = max(stages, key=lambda s: (max([TACTIC_WEIGHT.get(t, 0) for t in s["tactics"]] or [0]),
                                               s["top"]["similarity"]))
                r = dict(r, iocs=sorted({ip for s in stages for ip in s["iocs"]}),
                         campaign=next((s["campaign"] for s in stages if s["campaign"]), None))
            else:
                r = None
        else:
            r = self._match(f.observations, f.remote_ips)
            chain = [r["top"]["ext_id"]] if r else []
        if r is None:
            f.enrichment_status = "unmatched"
            f.enrichment = {"technique_id": None, "candidates": [], "engine": "built_in",
                            "severity_before": f.base_severity, "severity_after": f.severity}
            self.note(f"{f.finding_id}: no matching technique")
            return f
        top = r["top"]
        f.severity = severity_after_enrichment(f.base_severity if not f.members else
                                               SEVERITY[base_severity(f.confidence)],
                                               r["tactics"], bool(r["iocs"]), bool(f.members))
        vulns = [kg.g.nodes[v]["ext_id"] for v in kg.related(f"technique:{top['ext_id']}", "vulnerability")][:3]
        f.enrichment_status = "enriched"
        f.enrichment = {
            "technique_id": top["ext_id"], "technique_name": top["name"], "tactics": r["tactics"],
            "similarity": top["similarity"], "candidates": r["ranked"][:3], "matched_iocs": r["iocs"],
            "campaign": r["campaign"], "vulnerabilities": vulns, "engine": r["engine"], "chain": chain,
            "explanation": enrichment_text(top["ext_id"], top["name"], top["similarity"], r["iocs"], r["campaign"])
                           + (f"; attack chain {' -> '.join(chain)}" if len(chain) > 1 else ""),
            "severity_before": f.base_severity, "severity_after": f.severity,
        }
        kg.link_finding(f.finding_id, f.twin_ids, f"technique:{top['ext_id']}")
        self.note(f"{f.finding_id}: {top['ext_id']} {top['name']} ({top['similarity']:.2f})")
        self.log(self.name, "enriched", {"finding": f.finding_id, "technique": top["ext_id"],
                                         "similarity": top["similarity"], "chain": chain, "iocs": r["iocs"]})
        return f


# ====================================================================== PLAN
class ResponsePlanningAgent(Agent):
    """Mitigation choice is conditioned on the enriched technique (FR14)."""
    number, name, stage = 9, "Response Planning", "Plan"

    def plan(self, f: Finding) -> list[Action]:
        afddt = self.svc.afddt
        tech = f.enrichment.get("technique_id") or ""
        tactics = f.enrichment.get("tactics", [])
        devices = [t for t in f.twin_ids if t in afddt.devices]
        edges = [t for t in f.twin_ids if t in afddt.edges]
        gateway = next(iter(afddt.gateways))
        acts: list[Action] = []

        def dev_risk(d):
            return "high" if afddt.devices[d].critical else "low"

        def add(t, target, risk, desc, **detail):
            acts.append(Action(f.finding_id, t, target, risk, desc, dict(detail, technique=tech or None)))

        known_ioc = f.enrichment.get("matched_iocs", [])
        for ip in [i for i in known_ioc if i not in self.svc.blocked_ips]:
            add("block_source_ip", gateway, "low", f"Block known-bad address {ip} at the gateway", ip=ip)

        if f.enrichment_status != "enriched":
            for d in devices:
                add("rate_limit_device", d, "low", f"Rate-limit {d} while the anomaly is investigated")
            if not devices:
                add("monitor", f.twin_ids[0], "low", f"Heighten monitoring of {f.twin_ids[0]}")
        elif tech in ("T1498", "T1584.005", "T1071") or (tech == "T1499" and not edges):
            bots = [d for d in devices if afddt.devices[d].telemetry and
                    afddt.devices[d].telemetry[-1]["pkt_rate"] > 100] or devices
            for d in bots:
                add("isolate_device", d, dev_risk(d), f"Isolate {d}, which is generating flood traffic")
            add("segment_network", gateway, "high",
                f"Segment the IoT network: restrict outbound traffic at {gateway} ({tech} is a known botnet technique)")
        elif tech in ("T1110", "T1078"):
            if f.confidence >= 0.7 or not known_ioc:
                for d in devices:
                    add("isolate_device", d, dev_risk(d), f"Isolate {d} until its credentials are reset")
            # low-confidence guessing from a known address: blocking the source is the proportionate response
        elif tech in ("T1601", "T1495"):
            for d in devices:
                add("reflash_firmware", d, dev_risk(d), f"Re-flash {d} with known-good firmware")
        elif tech in ("T1499", "T1496"):
            for e in edges:
                add("throttle_edge", e, "low", f"Stop rogue jobs and throttle workload on {e}")
        elif tech in ("T1021", "T1570"):
            for d in devices:
                for e in edges:
                    if self.svc.kg.attached_edge(d) != e:
                        add("quarantine_link", e, dev_risk(d), f"Quarantine {d}'s connection to {e}", device=d)
            if not acts:
                for d in devices:
                    add("isolate_device", d, dev_risk(d), f"Isolate {d}")
        elif tech in ("T1557", "T1040"):
            add("reset_gateway_bindings", gateway, "high", "Reset gateway address bindings and enforce source validation")
        elif tech in ("T1595", "T1046", "T1190"):
            if not known_ioc:
                add("monitor", gateway, "low", "Heighten monitoring of scanned services")
        elif "impact" in tactics and devices:
            for d in devices:
                add("isolate_device", d, dev_risk(d), f"Isolate {d}")
        else:
            add("monitor", f.twin_ids[0], "low", f"Heighten monitoring of {f.twin_ids[0]}")
        f.actions = acts
        self.note(f"{f.finding_id}: {len(acts)} action(s) for {tech or 'unclassified anomaly'}")
        return acts

    @staticmethod
    def alternative(a: Action) -> Action | None:
        """Less disruptive option when simulation predicts disruption."""
        if a.type in ("isolate_device", "quarantine_link"):
            dev = a.detail.get("device", a.target)
            return Action(a.finding_id, "rate_limit_device", dev, "low",
                          f"Rate-limit {dev} instead of isolating it (critical device)",
                          {"technique": a.detail.get("technique")})
        return None


class MitigationSimulationAgent(Agent):
    number, name, stage = 10, "Mitigation Simulation", "Plan"

    def simulate(self, a: Action) -> Action:
        a.simulation = self.svc.ml.simulate({"type": a.type, "target": a.target, "detail": a.detail})
        self.note(f"{a.action_id} {a.type} on {a.target}: {a.simulation['note']}")
        return a


# ====================================================================== EXECUTE / LEARN
class ActionExecutionAgent(Agent):
    number, name, stage = 11, "Action Execution", "Execute"

    def __init__(self, svc, log, deployment):
        super().__init__(svc, log)
        self.deployment = deployment

    def execute(self, a: Action, approved_by: str = "Orchestrator (automatic)") -> Action:
        a.outcome = self.deployment.apply_mitigation(a.type, a.target, a.detail)
        a.status, a.approved_by = "executed", approved_by
        twin = self.svc.afddt.get(a.detail.get("device", a.target))
        if a.type in ("isolate_device", "quarantine_link") and twin is not None:
            twin.status = "isolated"
        elif a.type == "rate_limit_device" and twin is not None:
            twin.status = "rate_limited"
        elif a.type == "segment_network":
            next(iter(self.svc.afddt.gateways.values())).status = "segmented"
        elif a.type == "reflash_firmware" and twin is not None:
            twin.status = "normal"
        self.note(f"{a.action_id}: {a.outcome}")
        self.log(self.name, "executed", {"action": a.action_id, "type": a.type, "target": a.target,
                                         "outcome": a.outcome, "by": approved_by})
        return a


class LearningFeedbackAgent(Agent):
    number, name, stage = 12, "Learning/Feedback", "Learn"

    def __init__(self, svc, log):
        super().__init__(svc, log)
        self.fp_memory: set[tuple] = set()
        self.enrich_scored = 0
        self.enrich_correct = 0
        self.outcomes = defaultdict(int)

    def signature(self, f: Finding) -> tuple:
        devs = self.svc.afddt.devices
        classes = sorted({devs[t].device_class for t in f.twin_ids if t in devs})
        return (f.source_agent, f.enrichment.get("technique_id"), tuple(classes))

    def record(self, f: Finding) -> None:
        self.outcomes[f.decision] += 1
        if f.ground_truth and f.enrichment.get("technique_id") is not None:
            self.enrich_scored += 1
            if f.enrichment["technique_id"] in f.ground_truth:
                self.enrich_correct += 1
        self.note(f"{f.finding_id}: outcome '{f.decision}' recorded")

    def false_positive(self, f: Finding) -> str:
        """Operator says the finding was not an attack: remember the pattern and retrain the detector."""
        self.fp_memory.add(self.signature(f))
        msg = "pattern added to false-positive memory"
        devs = [t for t in f.twin_ids if t in self.svc.afddt.devices]
        retrained = []
        for d in devs:
            twin = self.svc.afddt.devices[d]
            if twin.telemetry:
                x = np.array([float(twin.telemetry[-1].get(k, 0)) for k in DEVICE_FEATURES])
                self.svc.ml.retrain_device(twin.device_class, x)
                retrained.append(twin.device_class)
        if retrained:
            msg += f"; {', '.join(sorted(set(retrained)))} model(s) retrained"
        self.note(f"{f.finding_id}: {msg}")
        self.log(self.name, "false_positive", {"finding": f.finding_id, "result": msg})
        return msg

    def confirm_technique(self, f: Finding, technique_id: str) -> str:
        """Operator confirms or corrects the technique: the example improves future matching."""
        self.svc.ml.add_enrichment_example(". ".join(f.observations), technique_id)
        msg = f"enrichment model updated with a confirmed {technique_id} example"
        self.note(f"{f.finding_id}: {msg}")
        self.log(self.name, "enrichment_feedback", {"finding": f.finding_id, "technique": technique_id})
        return msg

    def accuracy(self) -> float | None:
        return round(self.enrich_correct / self.enrich_scored, 3) if self.enrich_scored else None
