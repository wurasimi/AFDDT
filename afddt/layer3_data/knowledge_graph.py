"""Layer 3.2: Knowledge graph.

NetworkX stands in for a graph database.  The graph holds:

* twin nodes: one per device (leaf), edge/fog node, and the gateway (hub);
* relationship edges: device -> edge (attached_to), edge -> gateway (routes_via),
  plus observed connections that are NOT part of the normal topology;
* threat-intelligence nodes: techniques, indicators, vulnerabilities, campaigns
  (added by ti_ingest.py), with "indicates" / "uses" relationships;
* finding nodes linked to the twins they concern and the technique they matched.
"""
from __future__ import annotations

import threading

import networkx as nx


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.DiGraph()
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ twins
    def add_topology(self, topo: dict) -> None:
        gw = topo["gateway"]["id"]
        with self.lock:
            self.g.add_node(gw, kind="gateway", label=gw)
            for e in topo["edges"]:
                self.g.add_node(e["id"], kind="edge", label=e["id"])
                self.g.add_edge(e["id"], gw, rel="routes_via")
            for d in topo["devices"]:
                self.g.add_node(d["id"], kind="device", label=d["id"], device_class=d["class"],
                                critical=d["critical"])
                self.g.add_edge(d["id"], d["edge"], rel="attached_to")

    def attached_edge(self, device_id: str) -> str | None:
        for _, tgt, data in self.g.out_edges(device_id, data=True):
            if data.get("rel") == "attached_to":
                return tgt
        return None

    def devices_of_edge(self, edge_id: str) -> list[str]:
        return [s for s, _, d in self.g.in_edges(edge_id, data=True) if d.get("rel") == "attached_to"]

    def observe_connection(self, device_id: str, edge_id: str, tick: int) -> bool:
        """Record a device->edge connection; returns True if it is NOT a normal attachment."""
        if self.attached_edge(device_id) == edge_id:
            return False
        with self.lock:
            self.g.add_edge(device_id, edge_id, rel="observed_conn", tick=tick, anomalous=True)
        return True

    def twin_path(self, a: str, b: str) -> list[str]:
        """Shortest path between two twins over the topology (ignores TI and finding nodes)."""
        with self.lock:
            sub = self.g.subgraph([n for n, d in self.g.nodes(data=True)
                                   if d.get("kind") in ("device", "edge", "gateway")]).to_undirected()
            try:
                return nx.shortest_path(sub, a, b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []

    def dependents(self, twin_id: str) -> list[str]:
        """Twins that lose connectivity if this twin is cut off."""
        kind = self.g.nodes[twin_id].get("kind") if twin_id in self.g else None
        if kind == "gateway":
            return [n for n, d in self.g.nodes(data=True) if d.get("kind") in ("device", "edge")]
        if kind == "edge":
            return self.devices_of_edge(twin_id)
        return []

    # ------------------------------------------------------------------ threat intelligence
    def add_ti_object(self, node_id: str, kind: str, **attrs) -> None:
        with self.lock:
            self.g.add_node(node_id, kind=kind, **attrs)

    def add_relation(self, src: str, tgt: str, rel: str) -> None:
        with self.lock:
            if src in self.g and tgt in self.g:
                self.g.add_edge(src, tgt, rel=rel)

    def related(self, node_id: str, kind: str) -> list[str]:
        if node_id not in self.g:
            return []
        nbrs = set(self.g.successors(node_id)) | set(self.g.predecessors(node_id))
        return [n for n in nbrs if self.g.nodes[n].get("kind") == kind]

    # ------------------------------------------------------------------ findings
    def link_finding(self, finding_id: str, twin_ids: list[str], technique_node: str | None = None) -> None:
        with self.lock:
            self.g.add_node(finding_id, kind="finding")
            for t in twin_ids:
                if t in self.g:
                    self.g.add_edge(finding_id, t, rel="about")
            if technique_node and technique_node in self.g:
                self.g.add_edge(finding_id, technique_node, rel="matched")
            findings = [n for n, d in self.g.nodes(data=True) if d.get("kind") == "finding"]
            for old in findings[:-300]:          # keep the graph small
                self.g.remove_node(old)

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            out[d.get("kind", "?")] = out.get(d.get("kind", "?"), 0) + 1
        out["relationships"] = self.g.number_of_edges()
        return out
