"""Layer 7.2: the Orchestrator Agent and the MAPE-K loop (Section 3.7).

Each finding is carried through Analyze -> Plan -> decide -> Execute -> Learn
by a LangGraph state graph (if LangGraph is installed; otherwise the same
steps run in plain Python, in the same order).

The Orchestrator applies the autonomy policy of Table 3.8:

    1. pattern confirmed earlier as a false positive  -> dismiss
    2. simulation predicts disruption                 -> replan; hold if still disruptive
    3. high-risk action                               -> hold for human approval
    4. enriched severity high or critical             -> hold for human approval
    5. otherwise                                      -> execute automatically
    and, in addition, findings beyond local authority are escalated to the AIoTNDT.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from .agents import Action, Finding

try:
    from langgraph.graph import END, StateGraph
    HAVE_LANGGRAPH = True
except Exception:          # pragma: no cover - fallback path
    HAVE_LANGGRAPH = False


class LoopState(TypedDict, total=False):
    finding: Any
    actions: list


class Orchestrator:
    number, name, stage, twin = 13, "Orchestrator", "Supervise", "all"

    def __init__(self, agents: dict, svc, log):
        self.a = agents
        self.svc = svc
        self.log = log
        self.count = 0
        self.last = "idle"
        self.pending: dict[str, Action] = {}
        self.replanned = 0                      # unsafe actions replaced after simulation
        self.recent_techniques: list[tuple[int, str, str]] = []   # (tick, technique, device)
        self.engine_name = "LangGraph StateGraph" if HAVE_LANGGRAPH else "sequential (LangGraph not installed)"
        self.graph = self._build_graph() if HAVE_LANGGRAPH else None

    def info(self) -> dict:
        return {"no": 13, "name": self.name, "stage": self.stage, "twin": self.twin, "count": self.count,
                "last": self.last}

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        g = StateGraph(LoopState)
        g.add_node("enrich", self._enrich)
        g.add_node("plan", self._plan)
        g.add_node("simulate", self._simulate)
        g.add_node("decide", self._decide)
        g.add_node("learn", self._learn)
        g.set_entry_point("enrich")
        g.add_edge("enrich", "plan")
        g.add_edge("plan", "simulate")
        g.add_edge("simulate", "decide")
        g.add_edge("decide", "learn")
        g.add_edge("learn", END)
        return g.compile()

    def process(self, f: Finding, tick: int) -> Finding:
        started = time.perf_counter()
        ref = max([m.tick for m in f.members] or [f.tick])      # newest evidence behind this (re)processing
        f.wait_ticks = max(0, tick - ref)          # ticks between detection and processing (correlation hold)
        state: LoopState = {"finding": f, "actions": []}
        if self.graph is not None:
            self.graph.invoke(state)
        else:
            for step in (self._enrich, self._plan, self._simulate, self._decide, self._learn):
                state.update(step(state))
        f.status = "processed"
        f.process_ms = round((time.perf_counter() - started) * 1000, 2)   # enrich -> plan -> simulate -> decide -> learn
        return f

    # ------------------------------------------------------------------ nodes
    def _enrich(self, s: LoopState) -> dict:
        f = self.a["enrich"].enrich(s["finding"])
        f.latency_ms = round((time.perf_counter() - f.t0) * 1000, 2)   # event -> explained, enriched finding
        return {"finding": f}

    def _plan(self, s: LoopState) -> dict:
        return {"actions": self.a["plan"].plan(s["finding"])}

    def _simulate(self, s: LoopState) -> dict:
        final = []
        for a in s["actions"]:
            self.a["sim"].simulate(a)
            if a.simulation["disruption"]:
                alt = self.a["plan"].alternative(a)
                if alt is not None:
                    self.replanned += 1
                    self.a["sim"].simulate(alt)
                    alt.description += f" [replanned: {a.simulation['note']}]"
                    a = alt
            final.append(a)
        s["finding"].actions = final
        return {"actions": final}

    def _decide(self, s: LoopState) -> dict:
        f: Finding = s["finding"]
        tick = f.tick
        learn = self.a["learn"]
        if learn.signature(f) in learn.fp_memory:
            for a in f.actions:
                a.status = "dismissed"
            f.decision = "dismissed (known false positive)"
        else:
            decisions = []
            for a in f.actions:
                if a.simulation.get("disruption"):
                    a.status = "awaiting_approval"
                elif a.risk == "high":
                    a.status = "awaiting_approval"
                elif f.severity in ("high", "critical") and a.type not in ("block_source_ip", "monitor"):
                    a.status = "awaiting_approval"
                else:
                    self.a["exec"].execute(a)
                if a.status == "awaiting_approval":
                    dup = next((p for p in self.pending.values() if (p.type, p.target, p.detail.get("device"))
                                == (a.type, a.target, a.detail.get("device"))), None)
                    if dup is None:
                        self.pending[a.action_id] = a
                    else:
                        a.status = f"awaiting_approval (same as {dup.action_id})"
                decisions.append(a.status)
            if not f.actions:
                f.decision = "logged"
            elif all(d == "executed" for d in decisions):
                f.decision = "auto-executed"
            elif all(d.startswith("awaiting_approval") for d in decisions):
                f.decision = "awaiting approval"
            else:
                f.decision = "partly executed, partly awaiting approval"
            # beyond local authority -> escalate (Table 3.8, last row)
            tech = f.enrichment.get("technique_id")
            devs = [t for t in f.twin_ids if t in self.svc.afddt.devices]
            for d in devs:
                self.recent_techniques.append((tick, tech, d))
            self.recent_techniques = [r for r in self.recent_techniques if tick - r[0] <= 60]
            spread = {d for t_, te, d in self.recent_techniques if te == tech and tech}
            tiers = {self.svc.afddt.tier_of(t) for t in f.twin_ids}
            if f.severity == "critical" or len(tiers) >= 3 or len(spread) >= 3:
                f.escalated = True
                res = self.svc.connection.escalate_to_aiotndt(f)
                f.decision += "; escalated to AIoTNDT"
                self.log(self.name, "escalated", {"finding": f.finding_id, "result": res})
        for a in f.actions:
            self.svc.store.upsert_action(a)
        self.count += 1
        self.last = f"{f.finding_id}: {f.decision}"
        self.log(self.name, "decision", {"finding": f.finding_id, "decision": f.decision,
                                         "actions": [(a.type, a.target, a.status) for a in f.actions]})
        return {"finding": f}

    def _learn(self, s: LoopState) -> dict:
        self.a["learn"].record(s["finding"])
        self.svc.store.upsert_finding(s["finding"])
        return {}

    # ------------------------------------------------------------------ human decisions
    def approve(self, action_id: str, operator: str = "operator") -> Action | None:
        a = self.pending.pop(action_id, None)
        if a is not None:
            self.a["exec"].execute(a, approved_by=operator)
            self.svc.store.upsert_action(a)
            self.last = f"{action_id} approved by {operator}"
        return a

    def reject(self, action_id: str, operator: str = "operator") -> Action | None:
        a = self.pending.pop(action_id, None)
        if a is not None:
            a.status, a.approved_by = "rejected", operator
            self.svc.store.upsert_action(a)
            self.log(self.name, "rejected", {"action": action_id, "by": operator})
            self.last = f"{action_id} rejected by {operator}"
        return a
