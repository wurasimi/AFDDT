"""Wires the eight layers together and runs the system tick by tick.

One tick:
  Layer 1  the simulated deployment produces telemetry
  Layer 2  telemetry is published on the message bus
  Layer 3  subscribers sync the twins and write raw records
  Layer 7  monitor agents scan the twins (through Layer 6, using Layer 4/5)
           the fusion agent correlates findings across tiers
           each new finding goes through the Orchestrator's MAPE-K loop
  Layer 8  the dashboard reads snapshot() for display
"""
from __future__ import annotations

from pathlib import Path

import threading
import time
from collections import Counter, deque

import psutil

from .config import DB_PATH, SETTINGS, TI_DIR
from .layer1_physical.simulator import SCENARIOS, IoTDeployment
from .layer2_network.bus import MessageBus
from .layer3_data.knowledge_graph import KnowledgeGraph
from .layer3_data.store import DataStore
from .layer3_data.ti_ingest import load_bundles
from .layer4_models.detectors import ModelRegistry
from .layer4_models.embedding import Embedder
from .layer4_models.language_models import ReasoningEngine
from .layer4_models.simulation import MitigationSimulator
from .layer6_services.services import ServiceLayer
from .layer7_agents import agents as ag
from .layer7_agents.orchestrator import Orchestrator
from .twins import AFDDT, DeviceTwin, EdgeFogTwin, GatewayTwin


class Engine:
    def __init__(self, fresh_db: bool = True, quiet: bool = False, db_path=None):
        """db_path=":memory:" keeps the data store in memory, so tests and the evaluation never touch
        data/afddt.db, which the dashboard may have open."""
        say = (lambda *a: None) if quiet else print
        say("[AFDDT] Layer 1  building the simulated IoT deployment")
        self.deployment = IoTDeployment()
        topo = self.deployment.topology()

        say("[AFDDT] Layer 3  creating the twins, data store and knowledge graph")
        self.afddt = AFDDT()
        gw = topo["gateway"]["id"]
        self.afddt.gateways[gw] = GatewayTwin(gw)
        for e in topo["edges"]:
            self.afddt.edges[e["id"]] = EdgeFogTwin(e["id"], gw, [d["id"] for d in topo["devices"] if d["edge"] == e["id"]])
        for d in topo["devices"]:
            self.afddt.devices[d["id"]] = DeviceTwin(d["id"], d["class"], d["edge"], gw, d["ip"], d["critical"])
        self.store = DataStore(Path(db_path) if db_path else DB_PATH, fresh=fresh_db)
        self.kg = KnowledgeGraph()
        self.kg.add_topology(topo)

        say("[AFDDT] Layer 3  ingesting threat intelligence from data/ti")
        self.corpus = load_bundles(TI_DIR, self.kg)

        say(f"[AFDDT] Layer 4  training detection models on {SETTINGS.warmup_samples} benign samples")
        self.models = ModelRegistry(seed=SETTINGS.seed)
        classes = {d["id"]: d["class"] for d in topo["devices"]}
        self.models.train(self.deployment.benign_samples(SETTINGS.warmup_samples), classes)
        self.embedder = Embedder()
        self.reasoner = ReasoningEngine()
        self.simulator = MitigationSimulator(self.kg, self.afddt)

        say("[AFDDT] Layer 6  starting services")
        self.svc = ServiceLayer(self.afddt, self.store, self.kg, self.models, self.embedder, self.corpus,
                                self.reasoner, self.simulator)
        self.svc.gateway_hold = SETTINGS.gateway_hold_ticks
        self.svc.blocked_ips = self.deployment.blocked_ips

        say("[AFDDT] Layer 7  starting the twelve agents and the Orchestrator")
        cd = SETTINGS.finding_cooldown_ticks
        self.monitors = [ag.DeviceAuthFirmwareAgent(self.svc, self._log, cd),
                         ag.DeviceBehaviourAgent(self.svc, self._log, cd),
                         ag.ResourceExhaustionAgent(self.svc, self._log, cd),
                         ag.LateralMovementAgent(self.svc, self._log, cd),
                         ag.ProtocolTranslationAgent(self.svc, self._log, cd),
                         ag.TrafficBotnetAgent(self.svc, self._log, cd)]
        self.fusion = ag.CrossDomainFusionAgent(self.svc, self._log, SETTINGS.fusion_window_ticks,
                                                SETTINGS.gateway_hold_ticks)
        self.loop_agents = {"enrich": ag.ThreatEnrichmentAgent(self.svc, self._log),
                            "plan": ag.ResponsePlanningAgent(self.svc, self._log),
                            "sim": ag.MitigationSimulationAgent(self.svc, self._log),
                            "exec": ag.ActionExecutionAgent(self.svc, self._log, self.deployment),
                            "learn": ag.LearningFeedbackAgent(self.svc, self._log)}
        self.orchestrator = Orchestrator(self.loop_agents, self.svc, self._log)

        say("[AFDDT] Layer 2  subscribing the twins to the message bus")
        self.bus = MessageBus()
        self.bus.subscribe("afddt/device/", self._on_device)
        self.bus.subscribe("afddt/edge/", self._on_edge)
        self.bus.subscribe("afddt/gateway/", self._on_gateway)

        self.findings: list[ag.Finding] = []
        self.tick = 0
        self.tick_t0 = time.perf_counter()
        self.running = False
        self.lock = threading.RLock()
        self.started = time.time()
        self.tick_ms = deque(maxlen=50)
        self.msgs_last_tick = 0
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)

    # ------------------------------------------------------------------ logging
    def _log(self, agent: str, event: str, detail) -> None:
        self.store.audit(self.tick, agent, event, detail)

    # ------------------------------------------------------------------ Layer 2 -> 3 handlers
    def _on_device(self, topic: str, msg: dict) -> None:
        twin = self.afddt.devices[msg["twin_id"]]
        twin.sync(msg)
        self.store.write_telemetry(msg["tick"], "device", twin.twin_id, msg)

    def _on_edge(self, topic: str, msg: dict) -> None:
        self.afddt.edges[msg["twin_id"]].sync(msg)
        self.store.write_telemetry(msg["tick"], "edge", msg["twin_id"], msg)

    def _on_gateway(self, topic: str, msg: dict) -> None:
        self.afddt.gateways[msg["twin_id"]].sync(msg)
        self.store.write_telemetry(msg["tick"], "gateway", msg["twin_id"], msg)

    # ------------------------------------------------------------------ one tick
    def step(self) -> None:
        with self.lock:
            t_start = time.perf_counter()
            frame = self.deployment.step()                          # Layer 1
            self.tick = frame["tick"]
            before = self.bus.published
            for did, m in frame["devices"].items():                 # Layer 2
                self.bus.publish(f"afddt/device/{did}/telemetry", m)
            for eid, m in frame["edges"].items():
                self.bus.publish(f"afddt/edge/{eid}/telemetry", m)
            self.bus.publish(f"afddt/gateway/{frame['gateway']['twin_id']}/telemetry", frame["gateway"])
            self.msgs_last_tick = self.bus.published - before

            new: list[ag.Finding] = []                               # Layer 7: Monitor
            for agent in self.monitors:
                new.extend(agent.run(self.tick, t_start))
            gt = self.deployment.active_ground_truth()
            for f in new:
                if f.status == "new" and f.confidence < 0.95:     # give the fusion agent one tick to correlate
                    f.status, f.hold_until = "held", self.tick + 1
                f.ground_truth = sorted({t for tw in f.twin_ids for t in gt.get(tw, set())})
                self.findings.append(f)
                self.store.upsert_finding(f)

            recent = [f for f in self.findings if self.tick - f.tick <= SETTINGS.fusion_window_ticks]
            fused, released = self.fusion.run(self.tick, recent)    # Layer 7: Analyze (fusion)
            for f in fused:
                f.ground_truth = sorted({t for tw in f.twin_ids for t in gt.get(tw, set())})
                if f not in self.findings:
                    self.findings.append(f)

            for f in [f for f in self.findings if f.status == "new"] + released:
                if f.status == "new":
                    self.orchestrator.process(f, self.tick)          # Analyze -> Plan -> Execute -> Learn
            self.orchestrator.pending = {k: v for k, v in self.orchestrator.pending.items()
                                         if v.status == "awaiting_approval"}
            self._sync_status()
            if self.tick % 30 == 0:
                self.store.prune(300, self.tick)
            self.store.commit()
            self.tick_ms.append((time.perf_counter() - t_start) * 1000)

    def _sync_status(self) -> None:
        for d in self.deployment.devices.values():
            twin = self.afddt.devices[d.asset_id]
            if d.isolated:
                twin.status = "isolated"
            elif d.rate_limited:
                twin.status = "rate_limited"
            elif twin.status in ("isolated", "rate_limited"):
                twin.status = "normal"
        open_ = {t for f in self.findings if self.tick - f.tick <= 15 and f.decision != "dismissed (known false positive)"
                 for t in f.twin_ids}
        for tid, twin in list(self.afddt.devices.items()) + list(self.afddt.edges.items()):
            if twin.status in ("normal", "suspect"):
                twin.status = "suspect" if tid in open_ else "normal"
        gw = next(iter(self.afddt.gateways.values()))
        gw.status = "segmented" if self.deployment.gateway_segmented else ("suspect" if gw.twin_id in open_ else "normal")

    def close(self) -> None:
        self.running = False
        self.store.conn.close()

    # ------------------------------------------------------------------ run loop
    def run_forever(self) -> None:
        self.running = True
        while self.running:
            t = time.time()
            try:
                self.step()
            except Exception as exc:          # keep the twin alive; report the problem
                import traceback
                traceback.print_exc()
                self._log("engine", "error", str(exc))
            time.sleep(max(0.0, SETTINGS.tick_seconds - (time.time() - t)))

    # ------------------------------------------------------------------ operator commands
    def start_scenario(self, name: str, target: str | None = None) -> dict:
        with self.lock:
            sc = self.deployment.start_scenario(name, target)
            self._log("operator", "scenario", {"name": name, "targets": sc.targets})
            return {"scenario": name, "targets": sc.targets, "duration_ticks": sc.duration,
                    "description": SCENARIOS[name]}

    def approve(self, action_id: str) -> dict:
        with self.lock:
            a = self.orchestrator.approve(action_id)
            return {"ok": a is not None, "outcome": a.outcome if a else "not found"}

    def reject(self, action_id: str) -> dict:
        with self.lock:
            a = self.orchestrator.reject(action_id)
            return {"ok": a is not None}

    def feedback(self, finding_id: str, kind: str, technique: str | None = None) -> dict:
        with self.lock:
            f = next((x for x in self.findings if x.finding_id == finding_id), None)
            if f is None:
                return {"ok": False, "message": "finding not found"}
            learn = self.loop_agents["learn"]
            if kind == "false_positive":
                f.feedback = "false positive"
                return {"ok": True, "message": learn.false_positive(f)}
            tech = technique or f.enrichment.get("technique_id")
            if not tech:
                return {"ok": False, "message": "no technique to confirm"}
            f.feedback = f"confirmed {tech}"
            return {"ok": True, "message": learn.confirm_technique(f, tech)}

    def set_connectivity(self, online: bool) -> dict:
        self.reasoner.online = online
        self._log("operator", "connectivity", {"online": online})
        return {"online": online, "engine": self.reasoner.active_engine()}

    def reset(self) -> dict:
        with self.lock:
            self.deployment.restore()
            for t in list(self.afddt.devices.values()) + list(self.afddt.edges.values()):
                t.status = "normal"
            next(iter(self.afddt.gateways.values())).status = "normal"
            self.orchestrator.pending.clear()
            for m in self.monitors:
                m.last_report.clear()
                getattr(m, "reported_hash", {}).clear()
            return {"ok": True}

    def ask(self, question: str) -> dict:
        with self.lock:
            ctx = self._question_context()
        text, engine = self.svc.cognitive.answer(question, ctx)
        return {"answer": text, "engine": engine}

    def _question_context(self) -> dict:
        techs = Counter(f.enrichment.get("technique_id") for f in self.findings if f.enrichment.get("technique_id"))
        rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        done = [f for f in self.findings if f.status == "processed"]
        serious = sorted(done, key=lambda f: (rank.get(f.severity, 0), f.tick), reverse=True)
        return {
            "isolated": [d for d, t in self.afddt.devices.items() if t.status == "isolated"],
            "pending_approvals": [f"{a.action_id} {a.type} on {a.target}" for a in self.orchestrator.pending.values()],
            "techniques_seen": dict(techs),
            "escalations": len(self.svc.connection.escalations),
            "recent_findings": [self._describe(f) for f in done[-5:]],
            "most_serious": [self._describe(f) for f in serious[:3]],
            "open_findings": len(done),
        }

    @staticmethod
    def _describe(f) -> str:
        tags = [f.enrichment.get("technique_id") or "no ATT&CK match", f.severity]
        if f.decision:
            tags.append(f.decision)
        return f"{f.finding_id} {f.summary} [{', '.join(tags)}]"

    # ------------------------------------------------------------------ dashboard snapshot
    def finding_dict(self, f: ag.Finding, full: bool = False) -> dict:
        d = {"id": f.finding_id, "tick": f.tick, "ts": f.ts, "tier": f.tier, "agent": f.source_agent,
             "twins": f.twin_ids, "confidence": round(f.confidence, 2), "severity": f.severity,
             "base_severity": f.base_severity, "summary": f.summary, "status": f.status, "decision": f.decision,
             "technique": f.enrichment.get("technique_id"), "technique_name": f.enrichment.get("technique_name"),
             "similarity": f.enrichment.get("similarity"), "escalated": f.escalated, "latency_ms": f.latency_ms,
             "fused_from": f.fused_from, "feedback": f.feedback}
        if full:
            d.update({"observations": f.observations, "explanation": f.explanation, "enrichment": f.enrichment,
                      "remote_ips": f.remote_ips, "ground_truth": f.ground_truth,
                      "actions": [self.action_dict(a) for a in f.actions]})
        return d

    @staticmethod
    def action_dict(a: ag.Action) -> dict:
        return {"id": a.action_id, "finding": a.finding_id, "type": a.type, "target": a.target, "risk": a.risk,
                "description": a.description, "simulation": a.simulation, "status": a.status,
                "approved_by": a.approved_by, "outcome": a.outcome}

    def snapshot(self) -> dict:
        with self.lock:
            dep = self.deployment
            gw = next(iter(self.afddt.gateways.values()))
            gmsg = gw.traffic[-1] if gw.traffic else {}
            visible = [f for f in self.findings if f.status in ("processed", "held")]
            lat = [f.latency_ms for f in self.findings if f.latency_ms]
            store_counts = {t: self.store.count(t) for t in ("telemetry", "findings", "actions", "audit")}
            kgc = self.kg.counts()
            decisions = Counter(f.decision.split(";")[0] for f in self.findings if f.decision)
            layers = [
                {"no": "1", "name": "Physical", "items": [
                    f"{len(dep.devices)} devices, {len(dep.edges)} edge/fog nodes, 1 gateway (simulated)",
                    f"active attacks: {', '.join(s.name for s in dep.scenarios if s.active) or 'none'}"]},
                {"no": "2", "name": "Network", "items": [
                    f"bus: {self.bus.mode}", f"{self.msgs_last_tick} messages per tick, "
                                             f"{self.bus.published:,} total ({self.bus.bytes / 1e6:.1f} MB)"]},
                {"no": "3.1", "name": "Raw data", "items": [
                    f"{store_counts['telemetry']:,} telemetry rows kept (SQLite)",
                    f"{store_counts['findings']} findings, {store_counts['actions']} actions, "
                    f"{store_counts['audit']:,} audit entries"]},
                {"no": "3.2", "name": "Knowledge graph & RAG", "items": [
                    f"{kgc.get('device', 0)} device, {kgc.get('edge', 0)} edge, {kgc.get('gateway', 0)} gateway nodes",
                    f"{kgc.get('technique', 0)} techniques, {kgc.get('indicator', 0)} indicators, "
                    f"{kgc.get('vulnerability', 0)} CVEs, {kgc.get('campaign', 0)} campaigns",
                    f"{kgc.get('relationships', 0)} relationships"]},
                {"no": "4", "name": "Models", "items": [
                    f"{len(self.models.device)} device-class forests, edge z-score + graph model, 2 gateway forests",
                    f"embedding: {self.embedder.name}", f"reasoning: {self.reasoner.active_engine()}",
                    f"{self.simulator.runs} mitigation simulations, {self.models.retrain_count} retrainings"]},
                {"no": "5", "name": "Explainable AI", "items": [
                    f"{sum(1 for f in self.findings if f.explanation)} findings explained",
                    f"{sum(1 for f in self.findings if f.enrichment.get('explanation'))} enrichment explanations"]},
                {"no": "6", "name": "Services", "items": [f"{k}: {v:,} calls" for k, v in self.svc.calls.items()]},
                {"no": "7.1", "name": "AI agents", "items": [
                    f"12 agents, {sum(a.count for a in self.monitors)} monitor findings",
                    f"{self.fusion.count} fusions, {self.loop_agents['enrich'].count} enrichments"]},
                {"no": "7.2", "name": "Orchestrator", "items": [
                    f"loop engine: {self.orchestrator.engine_name}",
                    ", ".join(f"{k}: {v}" for k, v in decisions.items()) or "no decisions yet"]},
                {"no": "8", "name": "Application", "items": ["this dashboard", f"{len(self.orchestrator.pending)} "
                                                                               "actions awaiting approval"]},
            ]
            agents = [a.info() for a in self.monitors] + [self.fusion.info()] + \
                     [self.loop_agents[k].info() for k in ("enrich", "plan", "sim", "exec", "learn")] + \
                     [self.orchestrator.info()]
            return {
                "tick": self.tick, "uptime": round(time.time() - self.started),
                "online": self.reasoner.online, "engine": self.reasoner.active_engine(),
                "topology": {
                    "gateway": {"id": gw.twin_id, "status": gw.status},
                    "edges": [{"id": e.twin_id, "status": e.status,
                               "cpu": round(e.telemetry[-1]["cpu"]) if e.telemetry else 0}
                              for e in self.afddt.edges.values()],
                    "devices": [{"id": d.twin_id, "class": d.device_class, "edge": d.edge_id, "status": d.status,
                                 "critical": d.critical,
                                 "pkt_rate": round(d.telemetry[-1]["pkt_rate"], 1) if d.telemetry else 0}
                                for d in self.afddt.devices.values()],
                    "lateral": [[u, v] for u, v, dd in self.kg.g.edges(data=True) if dd.get("rel") == "observed_conn"
                                and self.tick - dd.get("tick", 0) <= 30],
                },
                "gateway_traffic": {"out_pkts": round(gmsg.get("out_pkts", 0)), "unique_dst": gmsg.get("unique_dst", 0),
                                    "spoofed": gmsg.get("spoofed_frames", 0)},
                "layers": layers,
                "agents": agents,
                "findings": [self.finding_dict(f) for f in visible[-40:]][::-1],
                "pending": [dict(self.action_dict(a), finding_summary=next(
                    (f.summary for f in self.findings if f.finding_id == a.finding_id), ""),
                    technique=a.detail.get("technique"))
                            for a in self.orchestrator.pending.values()],
                "audit": self.store.recent_audit(14),
                "metrics": {
                    "findings": len(visible), "fused": sum(1 for f in self.findings if f.tier == "cross-domain"),
                    "enriched": sum(1 for f in self.findings if f.enrichment_status == "enriched"),
                    "auto_executed": sum(1 for f in self.findings if f.decision.startswith("auto")),
                    "escalated": len(self.svc.connection.escalations),
                    "enrichment_accuracy": self.loop_agents["learn"].accuracy(),
                    "median_latency_ms": round(sorted(lat)[len(lat) // 2], 1) if lat else None,
                    "tick_ms": round(sum(self.tick_ms) / len(self.tick_ms), 1) if self.tick_ms else None,
                    "cpu_percent": self.proc.cpu_percent(None), "memory_mb": round(self.proc.memory_info().rss / 1e6),
                },
                "scenarios": SCENARIOS,
                "escalations": self.svc.connection.escalations[-5:][::-1],
            }
