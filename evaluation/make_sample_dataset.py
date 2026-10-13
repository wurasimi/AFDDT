"""Creates a SMALL SYNTHETIC dataset in the CICIoT2023 format.

It exists only so you can check that the evaluation pipeline runs before you
download the real dataset.  Its numbers are invented to resemble each attack
category and MUST NOT be reported in the dissertation.  The output file name
starts with SYNTHETIC so it cannot be mistaken for the real data.

Run:  python evaluation/make_sample_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ciciot import FEATURES

OUT = Path(__file__).resolve().parent.parent / "data" / "ciciot2023_sample"

LABELS = {
    "BenignTraffic": 6000, "DDoS-UDP_Flood": 900, "DDoS-SYN_Flood": 900, "DDoS-ICMP_Flood": 900,
    "DDoS-HTTP_Flood": 300, "DoS-UDP_Flood": 700, "DoS-SYN_Flood": 700, "DoS-HTTP_Flood": 300,
    "Mirai-udpplain": 800, "Mirai-greeth_flood": 600, "Recon-PortScan": 600, "Recon-OSScan": 400,
    "Recon-HostDiscovery": 400, "VulnerabilityScan": 300, "SqlInjection": 300, "CommandInjection": 300,
    "XSS": 250, "DictionaryBruteForce": 400, "MITM-ArpSpoofing": 500, "DNS_Spoofing": 400,
}


def rows(label, n, rng):
    d = {f: np.zeros(n) for f in FEATURES}
    N = lambda m, s: np.abs(rng.normal(m, s, n))
    # benign baseline
    d["flow_duration"] = N(8, 4); d["Header_Length"] = N(2e4, 1e4); d["Protocol Type"] = rng.choice([6, 17], n)
    d["Duration"] = N(64, 2); d["Rate"] = N(30, 15); d["Srate"] = d["Rate"]; d["ack_flag_number"] = rng.random(n) < .6
    d["syn_flag_number"] = rng.random(n) < .05; d["ack_count"] = N(.5, .3); d["syn_count"] = N(.2, .2)
    d["TCP"] = rng.random(n) < .6; d["UDP"] = 1 - d["TCP"]; d["HTTPS"] = rng.random(n) < .4
    d["DNS"] = rng.random(n) < .05; d["IPv"] = 1; d["LLC"] = 1
    d["Tot sum"] = N(5000, 2000); d["Min"] = N(60, 10); d["Max"] = N(1200, 300); d["AVG"] = N(500, 150)
    d["Std"] = N(300, 100); d["Tot size"] = N(500, 150); d["IAT"] = N(8e7, 2e7); d["Number"] = N(9, 2)
    d["Magnitue"] = N(30, 6); d["Radius"] = N(200, 80); d["Covariance"] = N(9e4, 3e4); d["Variance"] = N(.8, .2)
    d["Weight"] = N(140, 40)
    if label.startswith(("DDoS", "DoS", "Mirai")):
        scale = 4 if label.startswith("DDoS") or label.startswith("Mirai") else 1.5
        d["Rate"] = N(3000 * scale, 800); d["Srate"] = d["Rate"]; d["IAT"] = N(8e3, 4e3)
        d["Tot size"] = N(60, 6); d["AVG"] = N(60, 6); d["Std"] = N(3, 2); d["Max"] = N(70, 8); d["Min"] = N(55, 5)
        d["Variance"] = N(.05, .05); d["Magnitue"] = N(11, 1); d["Radius"] = N(4, 3)
        if "UDP" in label or "udp" in label or "greeth" in label:
            d["UDP"] = 1; d["TCP"] = 0; d["HTTPS"] = 0
        if "SYN" in label:
            d["syn_flag_number"] = 1; d["syn_count"] = N(2, .5); d["TCP"] = 1; d["UDP"] = 0; d["ack_flag_number"] = 0
        if "ICMP" in label:
            d["ICMP"] = 1; d["TCP"] = 0; d["UDP"] = 0; d["HTTPS"] = 0
        if "HTTP" in label:
            d["HTTP"] = 1; d["TCP"] = 1; d["UDP"] = 0; d["Rate"] = N(400, 150); d["Tot size"] = N(400, 80)
    elif label.startswith("Recon") or label == "VulnerabilityScan":
        d["syn_flag_number"] = rng.random(n) < .7; d["syn_count"] = N(1.2, .4); d["rst_count"] = N(3, 1)
        d["rst_flag_number"] = rng.random(n) < .5; d["Rate"] = N(120, 50); d["Tot size"] = N(58, 6)
        d["AVG"] = N(58, 6); d["Number"] = N(2, 1); d["flow_duration"] = N(.5, .3); d["Std"] = N(4, 2)
    elif label in ("SqlInjection", "CommandInjection", "XSS"):
        d["HTTP"] = 1; d["TCP"] = 1; d["UDP"] = 0; d["psh_flag_number"] = rng.random(n) < .8
        d["Tot size"] = N(900, 200); d["AVG"] = N(900, 200); d["Max"] = N(1500, 50); d["Rate"] = N(15, 8)
    elif label == "DictionaryBruteForce":
        d["SSH"] = rng.random(n) < .5; d["Telnet"] = 1 - d["SSH"]; d["TCP"] = 1; d["UDP"] = 0
        d["psh_flag_number"] = rng.random(n) < .7; d["Rate"] = N(60, 20); d["Tot size"] = N(110, 20)
        d["AVG"] = N(110, 20); d["flow_duration"] = N(2, 1)
    elif "Spoofing" in label:
        if "Arp" in label:
            d["ARP"] = 1; d["TCP"] = 0; d["UDP"] = 0; d["IPv"] = 0; d["Tot size"] = N(42, 2); d["AVG"] = N(42, 2)
        else:
            d["DNS"] = 1; d["UDP"] = 1; d["TCP"] = 0; d["Tot size"] = N(120, 20)
        d["Rate"] = N(80, 30)
    df = pd.DataFrame(d).astype(float)
    df["label"] = label
    return df


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    df = pd.concat([rows(l, n, rng) for l, n in LABELS.items()], ignore_index=True).sample(frac=1, random_state=1)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "SYNTHETIC_ciciot2023_format_sample.csv"
    df.to_csv(path, index=False)
    print(f"wrote {len(df):,} synthetic rows to {path}")
