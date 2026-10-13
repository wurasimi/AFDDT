"""CICIoT2023 helpers: column names, label mapping and ground truth (Table 3.11).

Put the dataset's CSV files (for example part-00000-...csv) in data/ciciot2023/.
The loader samples rows from every file so the evaluation runs on a laptop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
]

# Feature partition used to emulate what each tier can observe (Section 3.10.2, experiment E1)
DEVICE_VIEW = ["flow_duration", "Duration", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
               "Magnitue", "Radius", "Covariance", "Variance", "Weight", "Telnet", "SSH"]
GATEWAY_VIEW = [f for f in FEATURES if f not in DEVICE_VIEW]
DEFAULT_FEATURES = list(FEATURES)
DEVICE_SET = set(DEVICE_VIEW)   # host-side measurements; every other column is treated as gateway-observable

# Which Layer 7 monitoring agent is responsible for each CICIoT2023 category (Section 4.3):
# device-side flows go to the device agents, gateway-side flows to the gateway agents.
AGENT_OF_CATEGORY = {
    "BruteForce": "Device Authentication & Firmware Integrity",
    "Web": "Device Behaviour Anomaly",
    "Spoofing": "Protocol Translation Security",
    "DDoS": "Traffic & Botnet Detection", "DoS": "Traffic & Botnet Detection",
    "Mirai": "Traffic & Botnet Detection", "Recon": "Traffic & Botnet Detection",
}

CATEGORIES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Web", "BruteForce", "Spoofing"]


def category(label: str) -> str | None:
    l = label.strip()
    if l.lower().startswith("benign"):
        return "Benign"
    if l.startswith("DDoS"):
        return "DDoS"
    if l.startswith("DoS"):
        return "DoS"
    if l.startswith("Mirai"):
        return "Mirai"
    if l.startswith("Recon") or l == "VulnerabilityScan":
        return "Recon"
    if l in ("SqlInjection", "CommandInjection", "XSS", "Backdoor_Malware", "Uploading_Attack", "BrowserHijacking"):
        return "Web"
    if "BruteForce" in l:
        return "BruteForce"
    if "Spoofing" in l:
        return "Spoofing"
    return None


def ground_truth(label: str) -> set[str]:
    """ATT&CK techniques that count as a correct enrichment for this label (Table 3.11)."""
    c = category(label)
    if c in ("DDoS", "DoS"):
        return {"T1499", "T1498"} if ("HTTP" in label or "SlowLoris" in label) else {"T1498"}
    return {"Mirai": {"T1498"}, "Recon": {"T1595", "T1046"}, "Web": {"T1190"},
            "BruteForce": {"T1110"}, "Spoofing": {"T1557"}}.get(c, set())


def clean_label(name: str) -> str:
    """'SqlInjection.pcap' -> 'SqlInjection', 'DDoS-HTTP_Flood-.pcap.csv' -> 'DDoS-HTTP_Flood'."""
    n = str(name).strip()
    for suffix in (".csv", ".pcap"):
        if n.lower().endswith(suffix):
            n = n[: -len(suffix)]
    for suffix in (".csv", ".pcap"):
        if n.lower().endswith(suffix):
            n = n[: -len(suffix)]
    return n.strip(" -_")


def _set_feature_views(columns: list[str]) -> None:
    """Uses whichever CICIoT2023 feature columns the files actually contain.

    The dataset has been published in more than one layout (for example with
    flow_duration/Duration, or with Time_To_Live instead), so the feature list is
    taken from the data rather than fixed in advance.
    """
    global FEATURES, DEVICE_VIEW, GATEWAY_VIEW
    feats = [c for c in columns if c not in ("label", "category")]
    device = [c for c in feats if c in DEVICE_SET]
    gateway = [c for c in feats if c not in DEVICE_SET]
    if len(device) < 3 or len(gateway) < 3:
        raise ValueError(f"too few usable feature columns to form device and gateway views: {feats}")
    FEATURES, DEVICE_VIEW, GATEWAY_VIEW = feats, device, gateway


def load(folder: Path, rows_per_file: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Reads every CSV under `folder`, including sub-folders.

    The official download is organised as one folder per traffic type
    (Benign_Final or BenignTraffic.pcap, DDoS-UDP_Flood, SqlInjection.pcap, ...).
    If a file has no `label` column, the name of the folder it sits in is used as
    its label (or, for a file placed directly in `folder`, its own file name).
    """
    files = sorted(folder.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files in {folder}")
    parts = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if "label" not in df.columns:
            if "Label" in df.columns:
                df = df.rename(columns={"Label": "label"})
            else:
                df["label"] = clean_label(f.parent.name if f.parent != folder else f.name)
        df["label"] = df["label"].astype(str).map(clean_label)
        if len(df) > rows_per_file:
            df = df.sample(rows_per_file, random_state=seed)
        parts.append(df)
        print(f"  {f.parent.name}/{f.name}: {len(df):,} rows, label '{df['label'].iloc[0]}'")
    df = pd.concat(parts, ignore_index=True)
    df["category"] = df["label"].map(category)
    unknown = sorted(df.loc[df["category"].isna(), "label"].unique())
    if unknown:
        print(f"[load] ignored labels it does not recognise: {unknown}")
    df = df.dropna(subset=["category"])
    num = [c for c in df.columns if c not in ("label", "category")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    num = [c for c in num if df[c].notna().any()]
    df = df[num + ["label", "category"]].copy()
    df[num] = df[num].fillna(0)
    _set_feature_views(num)
    print(f"  {len(FEATURES)} feature columns used ({len(DEVICE_VIEW)} device view, {len(GATEWAY_VIEW)} gateway view)")
    missing = [c for c in DEFAULT_FEATURES if c not in FEATURES]
    if missing:
        print(f"  not in these files (layout differs from the original release): {missing}")
    return df
