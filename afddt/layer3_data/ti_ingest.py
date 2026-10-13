"""Layer 3.2: Threat-intelligence ingestion (Section 3.6.1).

Reads every STIX 2.1 bundle (*.json) in data/ti and loads four kinds of
objects into the knowledge graph and the retrieval corpus:

    attack-pattern  -> ATT&CK technique
    indicator       -> indicator of compromise (IP addresses are extracted)
    vulnerability   -> CVE entry
    campaign / intrusion-set -> campaign or threat actor

STIX is the format MISP, OpenCTI and TAXII servers exchange, so the same code
reads a file exported from any of them, or the full MITRE ATT&CK bundle that
scripts/update_ti.py downloads.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

IP_RE = re.compile(r"ipv4-addr:value\s*=\s*'([\d.]+)'")


@dataclass
class TIDoc:
    node_id: str
    kind: str              # technique | indicator | vulnerability | campaign | example
    ext_id: str            # e.g. T1110, CVE-2024-0001, the IP of an indicator
    name: str
    text: str
    tactics: list = field(default_factory=list)
    source: str = ""


@dataclass
class TICorpus:
    docs: list = field(default_factory=list)
    ioc_ips: dict = field(default_factory=dict)      # ip -> indicator node id
    files: list = field(default_factory=list)

    def techniques(self) -> dict:
        return {d.ext_id: d for d in self.docs if d.kind == "technique"}


def _ext_id(obj: dict) -> str:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") in ("mitre-attack", "cve") and ref.get("external_id"):
            return ref["external_id"]
    return obj.get("name", obj["id"])


def load_bundles(ti_dir: Path, kg=None) -> TICorpus:
    corpus = TICorpus()
    stix_to_node: dict[str, str] = {}
    relationships = []
    for path in sorted(ti_dir.glob("*.json")):
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[ti] skipped {path.name}: {exc}")
            continue
        objects = bundle.get("objects", bundle if isinstance(bundle, list) else [])
        corpus.files.append(path.name)
        for o in objects:
            t = o.get("type")
            if o.get("revoked") or o.get("x_mitre_deprecated"):
                continue
            if t == "attack-pattern":
                tid = _ext_id(o)
                node = f"technique:{tid}"
                tactics = [p["phase_name"] for p in o.get("kill_chain_phases", [])]
                doc = TIDoc(node, "technique", tid, o.get("name", tid),
                            f"{o.get('name', '')}. {o.get('description', '')}", tactics, path.name)
            elif t == "indicator":
                ips = IP_RE.findall(o.get("pattern", ""))
                node = f"indicator:{o['id'][-12:]}"
                doc = TIDoc(node, "indicator", ips[0] if ips else o.get("name", ""), o.get("name", "indicator"),
                            f"{o.get('name', '')}. {o.get('description', '')}", [], path.name)
                for ip in ips:
                    corpus.ioc_ips[ip] = node
            elif t == "vulnerability":
                cve = _ext_id(o)
                node = f"vulnerability:{cve}"
                doc = TIDoc(node, "vulnerability", cve, o.get("name", cve),
                            f"{o.get('name', '')}. {o.get('description', '')}", [], path.name)
            elif t in ("campaign", "intrusion-set"):
                node = f"campaign:{o['id'][-12:]}"
                doc = TIDoc(node, "campaign", o.get("name", ""), o.get("name", ""),
                            f"{o.get('name', '')}. {o.get('description', '')}", [], path.name)
            elif t == "relationship":
                relationships.append(o)
                continue
            else:
                continue
            stix_to_node[o["id"]] = node
            corpus.docs.append(doc)
            if kg is not None:
                kg.add_ti_object(node, doc.kind, label=doc.name, ext_id=doc.ext_id, tactics=doc.tactics)
    if kg is not None:
        for r in relationships:
            s, d = stix_to_node.get(r.get("source_ref")), stix_to_node.get(r.get("target_ref"))
            if s and d:
                kg.add_relation(s, d, r.get("relationship_type", "related-to"))
    return corpus
