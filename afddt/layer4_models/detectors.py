"""Layer 4: Detection models (one or two per twin type, Section 3.5).

All models are trained on benign telemetry before the system starts, so they
learn what normal looks like and flag departures from it.

* Device twin:   one Isolation Forest per device CLASS (not per device).
* Edge twin:     a resource-deviation model (per-node z-scores) and a
                 graph-pattern model (connections outside the normal topology).
* Gateway twin:  an Isolation Forest on protocol-translation events and an
                 Isolation Forest on volumetric/topological traffic features.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

DEVICE_FEATURES = ["auth_failures", "power_w", "pkt_rate", "pkt_size", "malformed", "unique_dst"]
EDGE_FEATURES = ["cpu", "mem", "jobs", "battery"]
GW_TRANSLATION_FEATURES = ["translation_errors", "spoofed_frames", "malformed_frames", "new_routes"]
GW_VOLUME_FEATURES = ["out_pkts_per_device", "unique_dst", "syn_ratio", "udp_ratio", "icmp_ratio",
                      "inbound_unique_ports"]


def vec(msg: dict, features: list[str]) -> np.ndarray:
    return np.array([float(msg.get(f, 0.0)) for f in features], dtype=float)


class BaselineModel:
    """An Isolation Forest plus the benign mean/std used for explanations."""

    def __init__(self, features: list[str], seed: int = 0):
        self.features = features
        self.model = IsolationForest(n_estimators=150, contamination=0.005, random_state=seed)
        self.mean = None
        self.std = None
        self.threshold = 0.0
        self.trained_on = 0
        self.buffer: list[np.ndarray] = []

    def fit(self, X: np.ndarray) -> "BaselineModel":
        self.model.fit(X)
        self.mean = X.mean(axis=0)
        # floor the spread so a signal that is constant in training (e.g. spoofed frames = 0)
        # still produces a finite, interpretable deviation
        self.std = np.maximum.reduce([X.std(axis=0), np.full(X.shape[1], 0.5), 0.1 * np.abs(self.mean)])
        scores = -self.model.decision_function(X)
        self.threshold = float(np.quantile(scores, 0.995))
        self.trained_on = len(X)
        self.buffer = list(X)
        return self

    def score(self, x: np.ndarray) -> float:
        """Anomaly score; > 0 means anomalous.

        Isolation Forest scores saturate for extreme outliers, so the forest's
        verdict is combined with the largest deviation from the benign baseline:
        a sample is anomalous if the forest flags it and some signal is at least
        4 SD out, or if any signal is at least 6 SD out.
        """
        forest = float(-self.model.decision_function(x.reshape(1, -1))[0] - self.threshold)
        z = float(np.max(np.abs(self.zscores(x))))
        return max(forest if z >= 4 else -1.0, (z - 6) / 10)

    def score_many(self, X: np.ndarray) -> np.ndarray:
        """Same as score() for many samples in one call (much faster)."""
        forest = -self.model.decision_function(X) - self.threshold
        z = np.max(np.abs((X - self.mean) / self.std), axis=1)
        return np.maximum(np.where(z >= 4, forest, -1.0), (z - 6) / 10)

    def zscores(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def add_benign(self, x: np.ndarray) -> None:
        """Learning/Feedback Agent: add a confirmed false positive to the benign set and retrain."""
        self.buffer.append(x)
        self.fit(np.array(self.buffer[-2000:]))


class ModelRegistry:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.device: dict[str, BaselineModel] = {}
        self.edge_mean: dict[str, np.ndarray] = {}
        self.edge_std: dict[str, np.ndarray] = {}
        self.gw_translation: BaselineModel | None = None
        self.gw_volume: BaselineModel | None = None
        self.retrain_count = 0

    def train(self, samples: list[dict], device_classes: dict[str, str]) -> None:
        by_class: dict[str, list] = {}
        edge_rows: dict[str, list] = {}
        gw_t, gw_v = [], []
        for s in samples:
            for did, m in s["devices"].items():
                by_class.setdefault(device_classes[did], []).append(vec(m, DEVICE_FEATURES))
            for eid, m in s["edges"].items():
                edge_rows.setdefault(eid, []).append(vec(m, EDGE_FEATURES))
            gw_t.append(vec(s["gateway"], GW_TRANSLATION_FEATURES))
            gw_v.append(vec(s["gateway"], GW_VOLUME_FEATURES))
        for cls, rows in by_class.items():
            self.device[cls] = BaselineModel(DEVICE_FEATURES, self.seed).fit(np.array(rows))
        for eid, rows in edge_rows.items():
            X = np.array(rows)
            self.edge_mean[eid] = X.mean(axis=0)
            self.edge_std[eid] = np.maximum(X.std(axis=0), 0.5)
        self.gw_translation = BaselineModel(GW_TRANSLATION_FEATURES, self.seed).fit(np.array(gw_t))
        self.gw_volume = BaselineModel(GW_VOLUME_FEATURES, self.seed).fit(np.array(gw_v))

    def edge_z(self, eid: str, msg: dict) -> np.ndarray:
        return (vec(msg, EDGE_FEATURES) - self.edge_mean[eid]) / self.edge_std[eid]

    def summary(self) -> list[dict]:
        out = [{"model": f"Isolation Forest ({cls})", "twin": "device", "trained_on": m.trained_on}
               for cls, m in self.device.items()]
        out += [{"model": "Resource-deviation (z-score)", "twin": "edge", "trained_on": len(self.edge_mean)},
                {"model": "Graph-pattern (topology)", "twin": "edge", "trained_on": "knowledge graph"},
                {"model": "Isolation Forest (translation)", "twin": "gateway", "trained_on": self.gw_translation.trained_on},
                {"model": "Isolation Forest (volumetric)", "twin": "gateway", "trained_on": self.gw_volume.trained_on}]
        return out
