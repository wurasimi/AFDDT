"""Layer 5: Explainable AI.

Every finding carries an explanation in the vocabulary of the detector that
produced it (Section 3.5):

* feature attribution: which measurements departed most from the learned
  benign baseline, in standard deviations (a z-score attribution, which needs
  no extra library; SHAP is applied in the evaluation, experiment E6, when the
  shap package is installed);
* graph-path evidence: the relationship in the knowledge graph that triggered
  a lateral-movement finding;
* enrichment explanation: which technique matched, at what similarity, and
  why.

`observations()` turns the top attributions into plain security language.
Those sentences are what the Threat Enrichment Agent matches against threat
intelligence, so enrichment depends on what was actually observed.
"""
from __future__ import annotations

import numpy as np

# feature -> (phrase when unusually HIGH, phrase when unusually LOW)
PHRASES = {
    "auth_failures": ("many repeated failed authentication login attempts guessing passwords", None),
    "power_w": ("unusual power draw", "power draw below normal"),
    "pkt_rate": ("very high packet rate flooding traffic", "packet rate dropped"),
    "pkt_size": ("unusually large packets", "small packets typical of floods"),
    "malformed": ("malformed packets", None),
    "unique_dst": ("traffic sent to many destinations", None),
    "cpu": ("CPU saturated and exhausted, host unresponsive", None),
    "mem": ("memory exhausted", None),
    "jobs": ("many unexpected running processes and jobs", None),
    "battery": (None, "battery and energy draining fast"),
    "translation_errors": ("protocol translation errors on forged frames", None),
    "spoofed_frames": ("spoofed addresses and forged frames, ARP poisoning, intercept traffic between devices", None),
    "malformed_frames": ("malformed frames at the gateway", None),
    "new_routes": ("unexpected new routing entries", None),
    "out_pkts_per_device": ("high volume flood of outbound packets", None),
    "syn_ratio": ("high share of TCP SYN probe packets", None),
    "udp_ratio": ("UDP flood traffic", None),
    "icmp_ratio": ("ICMP flood traffic", None),
    "inbound_unique_ports": ("scanning probes against many ports, port scan of exposed services", None),
}


def attribution(features: list[str], x: np.ndarray, mean: np.ndarray, std: np.ndarray, top: int = 3) -> list[dict]:
    z = (x - mean) / std
    order = np.argsort(-np.abs(z))[:top]
    return [{"feature": features[i], "value": round(float(x[i]), 2), "baseline": round(float(mean[i]), 2),
             "z": round(float(z[i]), 1)} for i in order if abs(z[i]) >= 3]


def observations(attr: list[dict]) -> list[str]:
    out = []
    for a in attr:
        hi, lo = PHRASES.get(a["feature"], (None, None))
        p = hi if a["z"] > 0 else lo
        if p:
            out.append(p)
    return out


def attribution_text(attr: list[dict]) -> str:
    return "; ".join(f"{a['feature']} = {a['value']} (normal {a['baseline']}, {a['z']:+.1f} SD)" for a in attr)


def enrichment_text(tech_id: str, tech_name: str, sim: float, iocs: list[str], campaign: str | None) -> str:
    s = f"matched ATT&CK {tech_id} ({tech_name}) at {sim:.2f} similarity"
    if iocs:
        s += f"; known indicator(s) {', '.join(iocs)}"
    if campaign:
        s += f", linked to campaign '{campaign}'"
    return s
