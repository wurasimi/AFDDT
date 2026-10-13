"""Checks that every layer works on your computer.  Takes about a minute.

Run:  python scripts/self_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPECT = {                     # scenario -> techniques that count as a correct classification
    "brute_force": {"T1110"},
    "firmware_tamper": {"T1601"},
    "resource_exhaustion": {"T1499", "T1496"},
    "lateral_movement": {"T1021", "T1570"},
    "protocol_spoofing": {"T1557"},
    "recon": {"T1595", "T1046"},
    "ddos_botnet": {"T1498"},
    "mirai_campaign": {"T1110", "T1021", "T1498"},
}


def main():
    ok = True
    t = time.time()
    from afddt.engine import Engine
    eng = Engine(db_path=":memory:")
    print(f"\n[1/3] Engine built in {time.time() - t:.1f} s")

    for _ in range(120):
        eng.step()
    fp = len(eng.findings)
    print(f"[2/3] 120 ticks of normal traffic: {fp} false alarm(s) {'PASS' if fp == 0 else 'CHECK'}")
    ok &= fp == 0

    print("[3/3] Attack scenarios")
    for sc, expected in EXPECT.items():
        eng.reset()
        for _ in range(25):
            eng.step()
        n0 = len(eng.findings)
        eng.start_scenario(sc)
        for _ in range(55):
            eng.step()
        new = [f for f in eng.findings[n0:] if f.status == "processed"]
        techs = {f.enrichment.get("technique_id") for f in new} - {None}
        hit = bool(techs & expected)
        ok &= hit
        acts = sum(len(f.actions) for f in new)
        print(f"   {sc:20s} findings {len(new):2d}  techniques {', '.join(sorted(techs)) or '-':22s} "
              f"actions {acts:2d}  {'PASS' if hit else 'FAIL'}")
    eng.close()
    print("\nAll layers working." if ok else "\nSome checks failed; see the lines marked FAIL or CHECK.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
