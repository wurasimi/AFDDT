"""Downloads current threat intelligence into data/ti (needs internet).

* MITRE ATT&CK Enterprise (STIX 2.1 bundle, about 40 MB) from MITRE's GitHub
  repository: replaces the starter technique summaries with MITRE's official
  descriptions for every technique.
* CISA Known Exploited Vulnerabilities catalog: converted into STIX
  vulnerability objects so CVE entries appear in the knowledge graph.
* Optionally, any TAXII 2.1 collection (for example from a MISP or OpenCTI
  server): set TAXII_URL to the collection's objects URL and, if needed,
  TAXII_USER and TAXII_PASSWORD.

Run:  python scripts/update_ti.py
Then restart the AFDDT; it loads every bundle in data/ti at start-up.
"""
import json
import os
import sys
import uuid
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "ti"
ATTACK_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def fetch_attack():
    print("Downloading MITRE ATT&CK Enterprise bundle ...")
    r = requests.get(ATTACK_URL, timeout=120)
    r.raise_for_status()
    bundle = r.json()
    keep = [o for o in bundle["objects"] if o.get("type") in ("attack-pattern", "intrusion-set", "campaign", "relationship")]
    (OUT / "mitre_attack_enterprise.json").write_text(json.dumps({"type": "bundle", "id": bundle["id"], "objects": keep}),
                                                      encoding="utf-8")
    seed = OUT / "attack_seed.json"
    if seed.exists():
        seed.rename(OUT / "attack_seed.json.disabled")   # official text now supersedes the summaries
    print(f"  saved {len(keep)} objects")


def fetch_kev(limit=400):
    print("Downloading CISA Known Exploited Vulnerabilities catalog ...")
    r = requests.get(KEV_URL, timeout=60)
    r.raise_for_status()
    vulns = r.json().get("vulnerabilities", [])[-limit:]
    objs = []
    for v in vulns:
        cve = v["cveID"]
        objs.append({
            "type": "vulnerability", "spec_version": "2.1",
            "id": f"vulnerability--{uuid.uuid5(uuid.NAMESPACE_URL, cve)}",
            "created": "2026-01-01T00:00:00.000Z", "modified": "2026-01-01T00:00:00.000Z",
            "name": cve,
            "description": f"{v.get('vendorProject', '')} {v.get('product', '')}: {v.get('vulnerabilityName', '')}. "
                           f"{v.get('shortDescription', '')}",
            "external_references": [{"source_name": "cve", "external_id": cve}],
        })
    (OUT / "cisa_kev.json").write_text(json.dumps({"type": "bundle", "id": "bundle--kev", "objects": objs}),
                                       encoding="utf-8")
    print(f"  saved {len(objs)} vulnerabilities")


def fetch_taxii():
    url = os.getenv("TAXII_URL")
    if not url:
        return
    print(f"Fetching TAXII collection {url} ...")
    auth = (os.getenv("TAXII_USER"), os.getenv("TAXII_PASSWORD")) if os.getenv("TAXII_USER") else None
    r = requests.get(url, headers={"Accept": "application/taxii+json;version=2.1"}, auth=auth, timeout=60)
    r.raise_for_status()
    objs = r.json().get("objects", [])
    (OUT / "taxii_feed.json").write_text(json.dumps({"type": "bundle", "id": "bundle--taxii", "objects": objs}),
                                         encoding="utf-8")
    print(f"  saved {len(objs)} objects")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for fn in (fetch_attack, fetch_kev, fetch_taxii):
        try:
            fn()
        except Exception as exc:
            ok = False
            print(f"  failed: {exc}")
    sys.exit(0 if ok else 1)
