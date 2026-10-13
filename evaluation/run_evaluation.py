"""Runs the experiments of Section 3.10 and writes tables and charts to results/.

    python evaluation/run_evaluation.py                       uses data/ciciot2023 (the real dataset)
    python evaluation/run_evaluation.py --data data/ciciot2023_sample   pipeline check on synthetic data

E1  cross-domain fusion: device view vs gateway view vs fused view of each flow (RQ1, RQ5)
E2  threat-intelligence enrichment: top-1 ATT&CK accuracy against Table 3.11, and how
    enrichment changes severity and the chosen mitigation (RQ2)
E3  detection latency and resource use of the live system (RQ3)
E4  autonomy policy: how the Orchestrator decided across repeated attack runs (RQ4)
E5  per-agent precision, recall and F1 (the dissertation's Table 4.1): device and gateway
    agents on CICIoT2023, edge agents and the fusion agent on the simulated testbed
E6  SHAP explanation of the fused detector (Layer 5), if the shap package is installed
"""
from __future__ import annotations

import argparse
import json
import warnings
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import ciciot  # noqa: E402

warnings.filterwarnings("ignore", category=ResourceWarning)
from afddt.config import RESULTS_DIR, SETTINGS, TI_DIR  # noqa: E402

INK, TWIN, GREY, WARN = "#1D2A33", "#0F7C80", "#9AA7A0", "#B87A07"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# CICIoT2023 feature -> plain-language observation when unusually high / low (Layer 5 vocabulary)
CIC_PHRASES = {
    "Rate": ("very high packet rate flooding traffic", None), "Srate": ("very high packet rate flooding traffic", None),
    "IAT": (None, "packets sent at very short intervals"),
    "Tot size": ("unusually large packets", "small packets typical of floods"),
    "AVG": ("unusually large packets", "small packets typical of floods"),
    "syn_flag_number": ("many TCP SYN packets", None), "syn_count": ("many TCP SYN packets", None),
    "rst_count": ("many connection resets, probing closed ports", None),
    "rst_flag_number": ("many connection resets, probing closed ports", None),
    "ICMP": ("ICMP flood traffic", None), "UDP": ("UDP flood traffic", None),
    "HTTP": ("web HTTP requests to an internet-facing application", None),
    "psh_flag_number": ("many pushed request payloads", None),
    "SSH": ("remote login attempts over SSH", None), "Telnet": ("remote login attempts over Telnet", None),
    "ARP": ("unusual ARP traffic, address resolution replies", None), "DNS": ("unusual DNS responses", None),
    "Number": (None, "very few packets per flow, many short probes"),
    "flow_duration": ("long-lived connections", "very short flows"),
}


def prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return round(p, 4), round(r, 4), round(f, 4), round(fp / max(1, fp + tn), 4)


# ---------------------------------------------------------------------- E1
def e1(train, test):
    rows, preds = [], {}
    views = {"Device view": ciciot.DEVICE_VIEW, "Gateway view": ciciot.GATEWAY_VIEW, "Fused (AFDDT)": ciciot.FEATURES}
    for name, feats in views.items():
        clf = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=0, class_weight="balanced")
        clf.fit(train[feats], train["category"])
        pred = clf.predict(test[feats])
        preds[name] = pred
        p, r, f, fpr = prf((test["category"] != "Benign").astype(int), (pred != "Benign").astype(int))
        _, _, mf, _ = precision_recall_fscore_support(test["category"], pred, average="macro", zero_division=0)
        rows.append({"view": name, "features": len(feats), "precision": p, "recall": r, "f1": f, "fpr": fpr,
                     "macro_f1_categories": round(mf, 4)})
        print(f"  E1 {name:16s} P={p:.4f} R={r:.4f} F1={f:.4f} FPR={fpr:.4f} macro-F1={mf:.4f}")
    per_class = []
    for name, pred in preds.items():
        p, r, f, s = precision_recall_fscore_support(test["category"], pred, labels=ciciot.CATEGORIES, zero_division=0)
        for c, pi, ri, fi, si in zip(ciciot.CATEGORIES, p, r, f, s):
            per_class.append({"view": name, "category": c, "precision": round(pi, 4), "recall": round(ri, 4),
                              "f1": round(fi, 4), "support": int(si)})
    return pd.DataFrame(rows), pd.DataFrame(per_class), preds["Fused (AFDDT)"]


# ---------------------------------------------------------------------- E2
def observations_for(row, mean, std, top=4):
    feats = [f for f in CIC_PHRASES if f in row.index]
    z = (row[feats] - mean[feats]) / std[feats]
    out = []
    for f in z.abs().sort_values(ascending=False).index[:top]:
        if abs(z[f]) < 3:
            continue
        hi, lo = CIC_PHRASES[f]
        p = hi if z[f] > 0 else lo
        if p and p not in out:
            out.append(p)
    return out


def e2(train, test, fused_pred, per_category=250):
    from afddt.layer3_data.knowledge_graph import KnowledgeGraph
    from afddt.layer3_data.ti_ingest import load_bundles
    from afddt.layer4_models.embedding import Embedder
    from afddt.layer4_models.language_models import ReasoningEngine
    from afddt.layer6_services.services import ServiceLayer
    from afddt.layer7_agents.agents import (SEVERITY, ThreatEnrichmentAgent, base_severity,
                                            severity_after_enrichment)

    kg = KnowledgeGraph()
    corpus = load_bundles(TI_DIR, kg)
    svc = ServiceLayer(None, None, kg, None, Embedder(), corpus, ReasoningEngine(), None)
    agent = ThreatEnrichmentAgent(svc, lambda *a: None)
    benign = train[train["category"] == "Benign"][ciciot.FEATURES]
    mean = benign.mean()
    # small floor: a protocol flag that is (almost) never set in benign traffic stands out strongly when set
    std = pd.Series(np.maximum(benign.std().values, 0.05) + 0.05 * np.abs(mean.values), index=mean.index)

    detected = test[(fused_pred != "Benign") & (test["category"] != "Benign")]
    sample = pd.concat([g.sample(min(len(g), per_category), random_state=0)
                        for _, g in detected.groupby("category")])
    cache, rows = {}, []
    for _, r in sample.iterrows():
        obs = observations_for(r[ciciot.FEATURES], mean, std)
        key = tuple(obs)
        if key not in cache:
            res = agent._match(obs, []) if obs else None
            cache[key] = res
        res = cache[key]
        tech = res["top"]["ext_id"] if res else None
        gt = ciciot.ground_truth(r["label"])
        base = SEVERITY[base_severity(0.8)]
        after = severity_after_enrichment(base, res["tactics"], False, False) if res else base
        rows.append({"label": r["label"], "category": r["category"], "observations": "; ".join(obs),
                     "technique": tech, "ground_truth": "/".join(sorted(gt)), "correct": tech in gt,
                     "similarity": res["top"]["similarity"] if res else 0.0,
                     "severity_without": base, "severity_with": after,
                     "action_without": "rate-limit and monitor", "action_with": action_family(tech)})
    df = pd.DataFrame(rows)
    df["has_evidence"] = df["observations"] != ""
    print(f"  E2 enrichment top-1 accuracy {df['correct'].mean():.4f} on {len(df)} detected attack flows; "
          f"coverage {df['has_evidence'].mean():.4f}; accuracy when evidence {df[df.has_evidence]['correct'].mean():.4f}")
    for c, row in e2_table(df).iterrows():
        print(f"     {c:12s} accuracy {row['accuracy']:.4f}  coverage {row['coverage']:.4f}  "
              f"accuracy when evidence {row['accuracy_when_evidence']:.4f}")
    return df


def e2_table(df):
    """Accuracy over all flows, share of flows with any observation, and accuracy on those flows."""
    t = df.groupby("category").agg(flows=("correct", "size"), accuracy=("correct", "mean"),
                                    coverage=("has_evidence", "mean"))
    t["accuracy_when_evidence"] = df[df.has_evidence].groupby("category")["correct"].mean()
    t["accuracy_when_evidence"] = t["accuracy_when_evidence"].fillna(0.0)
    return t.round(4)


def action_family(tech):
    if tech in ("T1498", "T1584.005", "T1071", "T1499"):
        return "isolate sources and segment network"
    if tech in ("T1110", "T1078"):
        return "block source and isolate device"
    if tech in ("T1557", "T1040"):
        return "reset gateway bindings"
    if tech in ("T1595", "T1046", "T1190"):
        return "block source and monitor"
    if tech in ("T1601", "T1495"):
        return "re-flash firmware"
    if tech in ("T1021", "T1570"):
        return "quarantine link"
    return "rate-limit and monitor"


# ---------------------------------------------------------------------- E3 / E4
def live_runs(repeats=3):
    from afddt.engine import Engine
    from afddt.layer1_physical.simulator import SCENARIOS
    lat, decisions, rows = [], Counter(), []
    lat_rows = []
    replanned = escalated = actions = 0
    action_status = Counter()
    per_run = []
    for rep in range(repeats):
        for sc in SCENARIOS:
            eng = Engine(quiet=True, db_path=":memory:")
            eng.deployment.pyrng.seed(100 + rep)
            for _ in range(30):
                eng.step()
            eng.start_scenario(sc)
            for _ in range(60):
                eng.step()
            for f in eng.findings:
                if f.status == "processed":
                    lat.append(f.latency_ms)
                    lat_rows.append({"scenario": sc, "finding": f.finding_id,
                                     "kind": "fused" if f.tier == "cross-domain" else "single-tier",
                                     "processing_ms": f.process_ms, "wait_ticks": f.wait_ticks,
                                     "live_latency_ms": round(f.wait_ticks * SETTINGS.tick_seconds * 1000
                                                              + f.process_ms, 1)})
                    decisions[f.decision.split(";")[0]] += 1
                    for a in f.actions:
                        actions += 1
                        action_status[a.status.split(" ")[0]] += 1
            tp, fp = Counter(), Counter()
            for f in {id(x): x for x in walk(eng.findings)}.values():
                (tp if f.ground_truth else fp)[f.source_agent] += 1
            per_run.append({"scenario": sc, "tp": tp, "fp": fp})
            replanned += eng.orchestrator.replanned
            escalated += len(eng.svc.connection.escalations)
            rows.append({"repeat": rep, "scenario": sc, "findings": sum(f.status == "processed" for f in eng.findings),
                         "tick_ms_mean": round(statistics.mean(eng.tick_ms), 1),
                         "memory_mb": round(eng.proc.memory_info().rss / 1e6),
                         "reasoning": eng.reasoner.active_engine()})
            eng.close()
    return (pd.DataFrame(lat_rows), decisions, action_status, replanned, escalated, actions, pd.DataFrame(rows),
            per_run)


# ---------------------------------------------------------------------- E5 / E6
def walk(findings):
    for f in findings:
        yield f
        yield from walk(f.members)


# Which agent the design (FR3-FR6) makes responsible for each simulated attack
EXPECTED_AGENTS = {
    "brute_force": ["Device Authentication & Firmware Integrity"],
    "firmware_tamper": ["Device Authentication & Firmware Integrity"],
    "resource_exhaustion": ["Resource Exhaustion Monitoring"],
    "lateral_movement": ["Lateral Movement Detection"],
    "protocol_spoofing": ["Protocol Translation Security"],
    "recon": ["Traffic & Botnet Detection"],
    "ddos_botnet": ["Traffic & Botnet Detection", "Device Behaviour Anomaly"],
    "mirai_campaign": ["Cross-Domain Threat Fusion"],
}


def e5_cic(test, fused_pred):
    """Per-agent scores on CICIoT2023: each agent is scored on the categories it is responsible for."""
    true_agent = test["category"].map(ciciot.AGENT_OF_CATEGORY).fillna("none").to_numpy()
    pred_agent = pd.Series(fused_pred).map(ciciot.AGENT_OF_CATEGORY).fillna("none").to_numpy()
    rows = []
    for agent in dict.fromkeys(ciciot.AGENT_OF_CATEGORY.values()):
        if not (true_agent == agent).any():
            continue          # no flows of this agent's categories in the files provided
        p, r, f = prf((true_agent == agent).astype(int), (pred_agent == agent).astype(int))[:3]
        rows.append({"agent": agent, "source": "CICIoT2023", "precision": p, "recall": r, "f1": f,
                     "support": int((true_agent == agent).sum())})
    return rows


def e5_live(per_run, benign_fp):
    """Per-agent scores on the simulated testbed.

    Precision: share of the agent's findings that point at a twin under attack (benign runs count as false).
    Recall: share of attack runs the agent is responsible for in which it raised at least one correct finding.
    """
    rows = []
    for agent in ["Resource Exhaustion Monitoring", "Lateral Movement Detection", "Cross-Domain Threat Fusion"]:
        tp = sum(r["tp"].get(agent, 0) for r in per_run)
        fp = sum(r["fp"].get(agent, 0) for r in per_run) + benign_fp.get(agent, 0)
        runs = [r for r in per_run if agent in EXPECTED_AGENTS[r["scenario"]]]
        hit = sum(1 for r in runs if r["tp"].get(agent, 0) > 0)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = hit / len(runs) if runs else 0.0
        f = 2 * p * rc / (p + rc) if p + rc else 0.0
        rows.append({"agent": agent, "source": "simulated testbed", "precision": round(p, 4), "recall": round(rc, 4),
                     "f1": round(f, 4), "support": len(runs)})
    return rows


def benign_runs(repeats, ticks=120):
    from afddt.engine import Engine
    fp = Counter()
    for rep in range(repeats):
        eng = Engine(quiet=True, db_path=":memory:")
        eng.deployment.pyrng.seed(500 + rep)
        for _ in range(ticks):
            eng.step()
        seen = {id(f): f for f in walk(eng.findings)}
        for f in seen.values():
            fp[f.source_agent] += 1
        eng.close()
    return fp


def chart_e5(df, path):
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(df))
    colors = [TWIN if s == "CICIoT2023" else "#5B4BB7" for s in df["source"]]
    ax.barh(y, df["f1"], color=colors)
    for i, v in enumerate(df["f1"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, df["agent"], fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1-score")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TWIN, label="CICIoT2023"), Patch(color="#5B4BB7", label="Simulated testbed")],
              frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("E5: F1-score per agent", loc="left", fontsize=10, pad=22)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def e6_shap(train, test, out, n=300):
    """Mean |SHAP value| per feature for the fused attack-vs-benign detector. Skipped if shap is missing."""
    try:
        import shap
    except ImportError:
        print("  E6 skipped: install it with  pip install shap  to add SHAP explanations")
        return None
    feats = ciciot.FEATURES
    clf = RandomForestClassifier(n_estimators=80, max_depth=12, n_jobs=-1, random_state=0, class_weight="balanced")
    clf.fit(train[feats], (train["category"] != "Benign").astype(int))
    sample = test[feats].sample(min(n, len(test)), random_state=0)
    vals = shap.TreeExplainer(clf).shap_values(sample)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(vals).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["mean_abs_shap"] = imp["mean_abs_shap"].round(5)
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh(top["feature"], top["mean_abs_shap"], color=TWIN)
    ax.set_xlabel("Mean |SHAP value| (contribution to the attack decision)")
    ax.set_title("E6: features that drive the fused detector (SHAP)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "e6_shap.png", dpi=200)
    plt.close(fig)
    imp.to_csv(out / "e6_shap.csv", index=False)
    print("  E6 top features: " + ", ".join(imp["feature"].head(5)))
    return imp


# ---------------------------------------------------------------------- charts
def chart_e1(df, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["f1"], 0.36, color=[GREY, GREY, TWIN], label="Attack-detection F1")
    ax.bar(x + 0.18, df["macro_f1_categories"], 0.36, color=[GREY, GREY, TWIN], alpha=0.55,
           label="Macro F1 across attack categories")
    for i, (a, b) in enumerate(zip(df["f1"], df["macro_f1_categories"])):
        ax.text(i - 0.18, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.18, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, df["view"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("E1: detection by view of the same flows", loc="left", fontsize=10, pad=22)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_e2(df, path):
    acc = df.groupby("category")["correct"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.barh(acc.index, acc.values, color=TWIN)
    for i, v in enumerate(acc.values):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Top-1 ATT&CK accuracy against Table 3.11")
    ax.set_title("E2: enrichment accuracy by attack category", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_e4(status, path):
    keys = [k for k in ("executed", "awaiting_approval", "dismissed", "rejected") if status.get(k)]
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.barh(keys, [status[k] for k in keys], color=[TWIN if k == "executed" else "#5B4BB7" for k in keys])
    for i, k in enumerate(keys):
        ax.text(status[k] + 0.3, i, str(status[k]), va="center", fontsize=8)
    ax.set_xlabel("Number of planned actions")
    ax.set_title("E4: how the Orchestrator handled planned actions", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "ciciot2023"))
    ap.add_argument("--rows-per-file", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--skip-live", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)
    synthetic = "sample" in data.name or not any(data.rglob("*.csv")) if data.exists() else True
    if not data.exists() or not any(data.rglob("*.csv")):
        print(f"No CSV files in {data}.\nDownload CICIoT2023 (see README, step 9) or run:\n"
              "  python evaluation/make_sample_dataset.py\n  python evaluation/run_evaluation.py --data data/ciciot2023_sample")
        sys.exit(1)
    out = RESULTS_DIR
    out.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"Loading {data} ...")
    df = ciciot.load(data, args.rows_per_file)
    print(f"  {len(df):,} flows, categories: {dict(Counter(df['category']))}")
    train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df["category"])

    print("E1 cross-domain fusion")
    e1_df, e1_cls, fused_pred = e1(train, test)
    e1_df.to_csv(out / "e1_detection.csv", index=False)
    e1_cls.to_csv(out / "e1_detection_per_category.csv", index=False)
    chart_e1(e1_df, out / "e1_detection.png")

    print("E2 threat-intelligence enrichment")
    e2_df = e2(train, test, fused_pred)
    e2_df.to_csv(out / "e2_enrichment.csv", index=False)
    chart_e2(e2_df, out / "e2_enrichment.png")
    sev = pd.crosstab(e2_df["severity_without"], e2_df["severity_with"])
    acts = pd.crosstab(e2_df["category"], e2_df["action_with"])

    live = None
    if not args.skip_live:
        print("E3/E4 live runs of every attack scenario")
        live = live_runs(args.repeats)
        lat, decisions, status, replanned, escalated, n_actions, runs, per_run = live
        runs.to_csv(out / "e3_runs.csv", index=False)
        lat.to_csv(out / "e3_latency.csv", index=False)
        pd.DataFrame(list(status.items()), columns=["status", "count"]).to_csv(out / "e4_policy.csv", index=False)
        chart_e4(status, out / "e4_policy.png")

    print("E5 per-agent scores")
    e5_rows = e5_cic(test, fused_pred)
    if live:
        e5_rows += e5_live(live[7], benign_runs(args.repeats))
    e5_df = pd.DataFrame(e5_rows)
    e5_df.to_csv(out / "e5_per_agent.csv", index=False)
    chart_e5(e5_df, out / "e5_per_agent.png")
    for r in e5_rows:
        print(f"  E5 {r['agent'][:38]:38s} {r['source']:17s} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")

    print("E6 SHAP explanation")
    e6_df = e6_shap(train, test, out)

    lines = ["# AFDDT evaluation results", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA.** These numbers come from the generated sample, not CICIoT2023. They only show "
                  "that the pipeline runs. Do not report them in the dissertation.", ""]
    lines += [f"Dataset: `{data}` ({len(df):,} flows; 70/30 stratified split). Run time {time.time() - t0:.0f} s.", "",
              "## E1: cross-domain fusion (RQ1, RQ5)", "", e1_df.to_markdown(index=False), "",
              "## E2: threat-intelligence enrichment (RQ2)", "",
              f"Top-1 accuracy: **{e2_df['correct'].mean():.4f}** on {len(e2_df)} detected attack flows. "
              f"Coverage (flows where at least one feature departed from the benign baseline by 3 standard "
              f"deviations, so there was something to match): {e2_df['has_evidence'].mean():.4f}. "
              f"Accuracy on those flows: {e2_df[e2_df.has_evidence]['correct'].mean():.4f}. "
              "A flow with no observation is left unclassified and counted as incorrect.", "",
              e2_table(e2_df).to_markdown(), "",
              "Severity without (rows) and with (columns) enrichment:", "", sev.to_markdown(), "",
              "Mitigation chosen with enrichment (without enrichment every flow gets 'rate-limit and monitor'):", "",
              acts.to_markdown(), ""]
    if live:
        lat, decisions, status, replanned, escalated, n_actions, runs, per_run = live
        by = lat.groupby("kind")[["processing_ms", "wait_ticks", "live_latency_ms"]].median().round(1)
        by.insert(0, "findings", lat.groupby("kind").size())
        p95 = lat.groupby("kind")[["processing_ms", "live_latency_ms"]].quantile(0.95).round(1)
        by["processing_ms_p95"], by["live_latency_ms_p95"] = p95["processing_ms"], p95["live_latency_ms"]
        lines += ["## E3: detection latency and resources (RQ3)", "",
                  "Medians per finding. processing_ms: compute time of the MAPE-K loop (enrich, plan, simulate, "
                  "decide, learn). wait_ticks: ticks a finding waited before processing, so that the fusion agent "
                  f"could correlate it. live_latency_ms: wait_ticks x the live tick interval ({SETTINGS.tick_seconds:g} s) "
                  "+ processing_ms, i.e. the time from the anomalous telemetry to the enriched, explained finding in "
                  "the live system. A fused finding is measured from the newest finding that joined it.", "",
                  by.to_markdown(), "",
                  f"Mean tick processing time {runs['tick_ms_mean'].mean():.1f} ms; peak memory "
                  f"{runs['memory_mb'].max()} MB; reasoning engine: {runs['reasoning'].iloc[0]}.", "",
                  "## E4: autonomy policy (RQ4)", "",
                  f"{n_actions} planned actions: " + ", ".join(f"{k} {v}" for k, v in status.items()) + ".",
                  f"Finding-level decisions: " + ", ".join(f"{k} {v}" for k, v in decisions.items()) + ".",
                  f"Unsafe actions replaced after simulation: {replanned}. Escalations to the AIoTNDT: {escalated}.", ""]
    lines += ["## E5: per-agent precision, recall and F1 (Table 4.1)", "",
              "CICIoT2023 rows: each agent is scored on the categories it is responsible for "
              "(BruteForce: device authentication; Web: device behaviour; Spoofing: protocol translation; "
              "DDoS, DoS, Mirai and Recon: traffic and botnet), using the fused detector's predictions. "
              "Simulated-testbed rows: CICIoT2023 has no separate edge/fog telemetry, so the edge agents and the "
              "fusion agent are scored on the scripted attacks, with benign-only runs counted for false positives. "
              "Support is flows for CICIoT2023 rows and attack runs for testbed rows.", "",
              e5_df.to_markdown(index=False), ""]
    if e6_df is not None:
        lines += ["## E6: SHAP explanation of the fused detector (Layer 5)", "",
                  "Mean absolute SHAP value per feature on a sample of test flows (attack vs benign model).", "",
                  e6_df.head(10).to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Open {out / 'summary.md'} and the PNG charts in {out}.")


if __name__ == "__main__":
    main()
