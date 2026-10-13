"""Layer 4: Mitigation simulation model (Section 3.7, agent 10).

Before any action reaches a physical asset, it is applied to a copy of the
twins' state and topology to predict its side effects: which twins would lose
connectivity, whether any critical device would be affected, and whether the
action would disrupt dependent services.
"""
from __future__ import annotations


class MitigationSimulator:
    def __init__(self, kg, afddt):
        self.kg = kg
        self.afddt = afddt
        self.runs = 0

    def simulate(self, action: dict) -> dict:
        self.runs += 1
        t, target = action["type"], action["target"]
        affected: list[str] = []
        if t in ("isolate_device", "reflash_firmware", "quarantine_link"):
            dev = action.get("detail", {}).get("device", target)
            affected = [dev]
        elif t == "segment_network":
            affected = [d for d in self.afddt.devices]           # outbound restricted for every device
        elif t == "throttle_edge":
            affected = self.kg.devices_of_edge(target)
        elif t in ("block_source_ip", "monitor", "notify", "rate_limit_device", "reset_gateway_bindings"):
            affected = []
        critical = [a for a in affected if getattr(self.afddt.devices.get(a), "critical", False)]
        # Disruption: cutting a critical device off entirely, or cutting off a large share of the deployment
        full_cut = t in ("isolate_device", "quarantine_link")
        disruption = bool(full_cut and critical)
        share = len(affected) / max(1, len(self.afddt.devices))
        if t == "segment_network":
            note = (f"outbound traffic of all {len(affected)} devices restricted; local services keep running, "
                    "external reporting delayed until the restriction is lifted")
        elif disruption:
            note = f"would cut off critical device(s) {', '.join(critical)}; dependent service interrupted"
        elif affected:
            note = f"affects {len(affected)} twin(s): {', '.join(affected[:4])}{'...' if len(affected) > 4 else ''}"
        else:
            note = "no twin loses connectivity"
        return {"affected": affected, "critical_affected": critical, "share_affected": round(share, 2),
                "disruption": disruption, "note": note}
