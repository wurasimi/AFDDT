"""Layer 6: Service layer.

Agents never touch the database, the graph or a model directly; they call
these four services (Section 3.4.3, second design principle).  Each call is
counted so the dashboard can show the layer working.

    6.1 ConnectionServices  escalation to the AIoTNDT, threat-intelligence refresh
    6.2 DataServices        twin state, graph queries, enrichment lookup
    6.3 MLServices          model scoring and retraining
    6.4 CognitiveServices   language-model reasoning and operator questions
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
import requests

from ..config import SETTINGS, TI_DIR
from ..layer3_data.ti_ingest import TIDoc, load_bundles
from ..layer4_models.detectors import DEVICE_FEATURES, EDGE_FEATURES, GW_TRANSLATION_FEATURES, GW_VOLUME_FEATURES, vec


class ConnectionServices:
    def __init__(self, parent):
        self.p = parent
        self.escalations: list[dict] = []

    def escalate_to_aiotndt(self, finding) -> str:
        self.p.calls["6.1 connection"] += 1
        payload = {"finding_id": finding.finding_id, "summary": finding.summary, "severity": finding.severity,
                   "technique": finding.enrichment.get("technique_id"), "twins": finding.twin_ids,
                   "ts": time.time()}
        if SETTINGS.aiotndt_url:
            try:
                requests.post(SETTINGS.aiotndt_url, json=payload, timeout=3)
                result = "sent to AIoTNDT"
            except Exception as exc:
                result = f"AIoTNDT unreachable ({exc.__class__.__name__}); queued"
        else:
            result = "queued for AIoTNDT (no cloud endpoint configured)"
        self.escalations.append({**payload, "result": result})
        return result

    def reload_threat_intelligence(self) -> int:
        self.p.calls["6.1 connection"] += 1
        self.p.corpus = load_bundles(TI_DIR, self.p.kg)
        self.p.refit_embeddings()
        return len(self.p.corpus.docs)


class DataServices:
    def __init__(self, parent):
        self.p = parent

    def twin(self, twin_id):
        self.p.calls["6.2 data"] += 1
        return self.p.afddt.get(twin_id)

    def latest(self, twin_id) -> dict | None:
        t = self.twin(twin_id)
        buf = t.traffic if hasattr(t, "traffic") else t.telemetry
        return buf[-1] if buf else None

    def window(self, twin_id, n: int) -> list[dict]:
        t = self.twin(twin_id)
        buf = t.traffic if hasattr(t, "traffic") else t.telemetry
        return list(buf)[-n:]

    def graph(self):
        self.p.calls["6.2 data"] += 1
        return self.p.kg

    def enrichment_lookup(self, text: str, k: int = 5) -> list[dict]:
        """Top-k threat-intelligence objects most similar to the text (Equation 3.1)."""
        self.p.calls["6.2 data"] += 1
        if self.p.doc_matrix is None or not text.strip():
            return []
        q = self.p.embedder.embed([text])[0]
        sims = self.p.doc_matrix @ q
        out = []
        for i in np.argsort(-sims)[: k * 3]:
            d: TIDoc = self.p.searchable[i]
            out.append({"node": d.node_id, "kind": d.kind, "ext_id": d.ext_id, "name": d.name,
                        "similarity": round(float(sims[i]), 3), "tactics": d.tactics})
        return out

    def indicators_for(self, ips: list[str]) -> list[tuple[str, str]]:
        self.p.calls["6.2 data"] += 1
        return [(ip, self.p.corpus.ioc_ips[ip]) for ip in ips if ip in self.p.corpus.ioc_ips]


class MLServices:
    def __init__(self, parent):
        self.p = parent

    def device_model(self, device_class):
        self.p.calls["6.3 ai/ml/sim"] += 1
        return self.p.models.device[device_class]

    def score_device(self, device_class: str, msg: dict) -> tuple[float, np.ndarray]:
        m = self.device_model(device_class)
        x = vec(msg, DEVICE_FEATURES)
        return m.score(x), x

    def score_devices(self, device_class: str, msgs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Score every device of one class in a single model call."""
        m = self.device_model(device_class)
        X = np.array([vec(msg, DEVICE_FEATURES) for msg in msgs])
        return m.score_many(X), X

    def edge_z(self, edge_id: str, msg: dict) -> np.ndarray:
        self.p.calls["6.3 ai/ml/sim"] += 1
        return self.p.models.edge_z(edge_id, msg)

    def score_gateway_translation(self, msg: dict):
        self.p.calls["6.3 ai/ml/sim"] += 1
        x = vec(msg, GW_TRANSLATION_FEATURES)
        return self.p.models.gw_translation.score(x), x

    def score_gateway_volume(self, msg: dict):
        self.p.calls["6.3 ai/ml/sim"] += 1
        x = vec(msg, GW_VOLUME_FEATURES)
        return self.p.models.gw_volume.score(x), x

    def simulate(self, action: dict) -> dict:
        self.p.calls["6.3 ai/ml/sim"] += 1
        return self.p.simulator.simulate(action)

    def retrain_device(self, device_class: str, x: np.ndarray) -> None:
        self.p.calls["6.3 ai/ml/sim"] += 1
        self.p.models.device[device_class].add_benign(x)
        self.p.models.retrain_count += 1

    def add_enrichment_example(self, text: str, technique_id: str) -> None:
        """Learning loop for enrichment: a confirmed (text, technique) pair joins the retrieval corpus."""
        self.p.calls["6.3 ai/ml/sim"] += 1
        tech = self.p.corpus.techniques().get(technique_id)
        name = tech.name if tech else technique_id
        tactics = tech.tactics if tech else []
        doc = TIDoc(f"example:{len(self.p.corpus.docs)}", "example", technique_id, name, text, tactics, "learned")
        self.p.corpus.docs.append(doc)
        self.p.refit_embeddings()
        self.p.models.retrain_count += 1


class CognitiveServices:
    SYSTEM = ("You are the reasoning service of an IoT security digital twin. Answer briefly and only from the "
              "information given.")

    def __init__(self, parent):
        self.p = parent

    def choose_technique(self, observations: list[str], candidates: list[dict]) -> tuple[str | None, str]:
        """Ask the language model to pick the best-matching technique; returns (technique_id, engine)."""
        self.p.calls["6.4 cognitive"] += 1
        if not candidates:
            return None, "built_in"
        lines = "\n".join(f"- {c['ext_id']} {c['name']} (similarity {c['similarity']})" for c in candidates)
        prompt = (f"Observed behaviour: {'; '.join(observations)}.\nCandidate MITRE ATT&CK techniques:\n{lines}\n"
                  "Reply with only the ID of the single technique that best explains the observed behaviour.")
        text, engine = self.p.reasoner.complete(prompt, self.SYSTEM, max_tokens=12)
        if text:
            for c in candidates:
                if c["ext_id"] in text:
                    return c["ext_id"], engine
        return None, engine

    def answer(self, question: str, context: dict) -> tuple[str, str]:
        self.p.calls["6.4 cognitive"] += 1
        prompt = f"System state (JSON): {json.dumps(context, default=str)[:6000]}\n\nOperator question: {question}"
        text, engine = self.p.reasoner.complete(prompt, self.SYSTEM, max_tokens=250)
        if text:
            return text.strip(), engine
        return self._built_in_answer(question.lower(), context), engine

    @staticmethod
    def _built_in_answer(q: str, c: dict) -> str:
        if any(w in q for w in ("serious", "severe", "worst", "critical", "danger", "priority", "urgent", "biggest")):
            top = c.get("most_serious", [])
            if not top:
                return "No threats have been found yet."
            ans = f"The most serious threat is {top[0]}."
            if len(top) > 1:
                ans += " Next: " + "; ".join(top[1:]) + "."
            return ans
        if "isolat" in q or "quarantin" in q:
            iso = c["isolated"]
            return f"Isolated devices: {', '.join(iso)}." if iso else "No device is isolated."
        if "approv" in q or "pending" in q or "waiting" in q:
            pend = c["pending_approvals"]
            return (f"{len(pend)} action(s) await approval: " + "; ".join(pend)) if pend else "Nothing awaits approval."
        if "technique" in q or "att&ck" in q or "attack" in q:
            t = c["techniques_seen"]
            return ("Techniques matched so far: " + ", ".join(f"{k} ({v})" for k, v in t.items())) if t else \
                "No technique has been matched yet."
        if "escalat" in q or "cloud" in q:
            return f"{c['escalations']} finding(s) escalated to the AIoTNDT."
        if "finding" in q or "alert" in q or "threat" in q:
            f = c["recent_findings"]
            return ("Most recent findings: " + "; ".join(f)) if f else "No findings yet."
        return ("I can answer questions about the most serious threat, isolated devices, pending approvals, matched "
                "techniques, escalations and recent findings. Install a local model with Ollama for open-ended questions.")


class ServiceLayer:
    def __init__(self, afddt, store, kg, models, embedder, corpus, reasoner, simulator):
        self.afddt, self.store, self.kg, self.models = afddt, store, kg, models
        self.embedder, self.corpus, self.reasoner, self.simulator = embedder, corpus, reasoner, simulator
        self.calls: Counter = Counter({"6.1 connection": 0, "6.2 data": 0, "6.3 ai/ml/sim": 0, "6.4 cognitive": 0})
        self.searchable: list[TIDoc] = []
        self.doc_matrix = None
        self.connection = ConnectionServices(self)
        self.data = DataServices(self)
        self.ml = MLServices(self)
        self.cognitive = CognitiveServices(self)
        self.refit_embeddings()

    def refit_embeddings(self) -> None:
        self.searchable = [d for d in self.corpus.docs if d.kind in ("technique", "example")]
        texts = [d.text for d in self.searchable]
        if texts:
            self.embedder.fit([d.text for d in self.corpus.docs])
            self.doc_matrix = self.embedder.embed(texts)
