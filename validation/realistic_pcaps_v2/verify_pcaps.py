#!/usr/bin/env python3
"""
Standalone verifier for the realistic-PCAP v2 (diversified) corpus.

No Django, no model. Reads ``MANIFEST.csv`` and independently re-verifies every
referenced PCAP with scapy:

  * exists, readable, has packets
  * every IP packet is inside the loopback net 127.0.0.0/8 (no external host)
  * the manifest's expected host and FTP control port actually appear
  * no non-TCP (UDP) payload
  * finite, non-negative capture duration

Also checks manifest<->disk agreement. Exits non-zero on any failure, so it can
gate a pre-retraining check. Labels are read from the manifest (which records the
capture scenario/folder), never inferred from a model.

    python3 validation/realistic_pcaps_v2/verify_pcaps.py [dir]
"""

from __future__ import annotations

import csv
import ipaddress
import math
import sys
from pathlib import Path

LOOPBACK = ipaddress.ip_network("127.0.0.0/8")


def _port_from_dest(dest: str) -> int:
    try:
        return int(dest.rsplit(":", 1)[1])
    except Exception:  # noqa: BLE001
        return 21


def verify_one(pcap: Path, host: str, port: int) -> dict:
    from scapy.all import rdpcap, TCP, UDP, IP

    r = {"pcap": pcap.name, "exists": pcap.is_file(), "readable": False, "packets": 0,
         "loopback_only": True, "host_present": False, "port_ok": False,
         "tcp_only": True, "duration_s": 0.0, "duration_valid": False, "errors": []}
    if not r["exists"]:
        r["errors"].append("missing"); return r
    try:
        pkts = rdpcap(str(pcap)); r["readable"] = True
    except Exception as e:  # noqa: BLE001
        r["errors"].append(f"unreadable: {e}"); return r
    r["packets"] = len(pkts)
    if not pkts:
        r["errors"].append("no packets"); return r

    times = []
    for pk in pkts:
        times.append(float(pk.time))
        if IP in pk:
            for a in (pk[IP].src, pk[IP].dst):
                if ipaddress.ip_address(a) not in LOOPBACK:
                    r["loopback_only"] = False
            if host in (pk[IP].src, pk[IP].dst):
                r["host_present"] = True
        if TCP in pk:
            if port in (pk[TCP].sport, pk[TCP].dport):
                r["port_ok"] = True
        elif UDP in pk:
            r["tcp_only"] = False
    if times:
        d = max(times) - min(times)
        r["duration_s"] = round(d, 4)
        r["duration_valid"] = d >= 0 and math.isfinite(d)

    if not r["loopback_only"]:
        r["errors"].append("non-loopback address present")
    if not r["host_present"]:
        r["errors"].append(f"expected host {host} absent")
    if not r["port_ok"]:
        r["errors"].append(f"no traffic on port {port}")
    if not r["tcp_only"]:
        r["errors"].append("non-TCP payload present")
    if not r["duration_valid"]:
        r["errors"].append("invalid duration")
    return r


def main(argv=None) -> int:
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent
    manifest = root / "MANIFEST.csv"
    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("ERROR: empty manifest", file=sys.stderr)
        return 2

    on_disk = {p.name for p in root.rglob("*.pcap")}
    # manifest stores only the label folder + filename via capture_id; match by capture_id
    missing, n_ok = [], 0
    print(f"Verifying {len(rows)} captures under {root}\n")
    print(f"{'capture':12} {'label':16} {'host':10} {'port':>5} {'pkts':>5} {'dur_s':>7}  status")
    for r in rows:
        folder = "benign" if r["label"] == "Benign" else "ftp_bruteforce"
        matches = list((root / folder).glob(f"{r['capture_id']}_*.pcap"))
        if not matches:
            missing.append(r["capture_id"])
            print(f"{r['capture_id']:12} {r['label']:16} MISSING")
            continue
        pcap = matches[0]
        host = r["source"]
        port = _port_from_dest(r["destination"])
        res = verify_one(pcap, host, port)
        ok = (res["exists"] and res["readable"] and res["packets"] > 0
              and res["loopback_only"] and res["host_present"] and res["port_ok"]
              and res["tcp_only"] and res["duration_valid"])
        n_ok += ok
        status = "ok" if ok else "FAIL: " + "; ".join(res["errors"])
        print(f"{r['capture_id']:12} {r['label']:16} {host:10} {port:5} "
              f"{res['packets']:5} {res['duration_s']:7.2f}  {status}")

    n_manifest_pcaps = sum(1 for r in rows)
    orphan = on_disk - {m.name for r in rows for m in
                        (root / ("benign" if r["label"] == "Benign" else "ftp_bruteforce"))
                        .glob(f"{r['capture_id']}_*.pcap")}
    if orphan:
        print(f"\nERROR: PCAPs on disk not in manifest: {sorted(orphan)}", file=sys.stderr)
    print(f"\n{n_ok}/{len(rows)} captures verified OK")
    return 0 if (n_ok == len(rows) and not missing and not orphan) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
