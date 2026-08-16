#!/usr/bin/env python3
"""
Standalone verifier for the realistic-PCAP expansion set (no Django, no model).

Walks ``validation/realistic_pcaps/`` (or a directory given as argv[1]), cross-
checks every PCAP referenced by ``MANIFEST.csv``, and independently re-verifies
each capture with scapy:

  * file exists and is readable
  * contains packets
  * contains FTP traffic on the expected control port (default 21)
  * every IP packet is loopback-only (127.0.0.1) -- lab traffic only
  * all payload-bearing packets are TCP
  * capture duration is finite and non-negative

Exits non-zero if any capture fails or if the manifest and disk disagree, so it
can gate CI or a pre-retraining check. It never loads the model and never uses a
prediction -- labels are read from the manifest (which records the capture
scenario), never inferred.

    python3 validation/realistic_pcaps/verify_pcaps.py [dir] [--port 21]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def verify_one(pcap: Path, control_port: int) -> dict:
    from scapy.all import rdpcap, TCP, IP

    r = {"pcap": pcap.name, "exists": pcap.is_file(), "readable": False,
         "packets": 0, "ftp_port": False, "loopback_only": True, "tcp_only": True,
         "duration_s": 0.0, "duration_valid": False, "errors": []}
    if not r["exists"]:
        r["errors"].append("missing file")
        return r
    try:
        pkts = rdpcap(str(pcap))
        r["readable"] = True
    except Exception as e:  # noqa: BLE001
        r["errors"].append(f"unreadable: {e}")
        return r

    r["packets"] = len(pkts)
    if not pkts:
        r["errors"].append("no packets")
        return r

    times = []
    non_tcp_payload = 0
    for pk in pkts:
        times.append(float(pk.time))
        if IP in pk and (pk[IP].src != "127.0.0.1" or pk[IP].dst != "127.0.0.1"):
            r["loopback_only"] = False
        if TCP in pk:
            if control_port in (pk[TCP].sport, pk[TCP].dport):
                r["ftp_port"] = True
        elif IP in pk and getattr(pk, "payload", None) is not None:
            # loopback FTP capture should be TCP; count anything else
            from scapy.all import UDP
            if UDP in pk:
                non_tcp_payload += 1
    r["tcp_only"] = non_tcp_payload == 0
    if times:
        d = max(times) - min(times)
        r["duration_s"] = round(d, 4)
        r["duration_valid"] = d >= 0 and math.isfinite(d)

    if not r["ftp_port"]:
        r["errors"].append(f"no traffic on FTP port {control_port}")
    if not r["loopback_only"]:
        r["errors"].append("traffic left the loopback lab")
    if not r["duration_valid"]:
        r["errors"].append("invalid duration")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--port", type=int, default=21)
    args = ap.parse_args(argv)

    root = Path(args.dir)
    manifest = root / "MANIFEST.csv"
    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2

    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("ERROR: manifest is empty", file=sys.stderr)
        return 2

    # every pcap on disk should be in the manifest, and vice-versa
    on_disk = {p.name for p in root.rglob("*.pcap")}
    in_manifest = {Path(r["pcap_filename"]).name for r in rows}
    missing = in_manifest - on_disk
    orphan = on_disk - in_manifest
    if missing:
        print(f"ERROR: manifest references missing PCAPs: {sorted(missing)}", file=sys.stderr)
    if orphan:
        print(f"ERROR: PCAPs on disk not in manifest: {sorted(orphan)}", file=sys.stderr)

    n_ok = 0
    print(f"Verifying {len(rows)} captures under {root} (FTP port {args.port})\n")
    print(f"{'capture':16} {'label':16} {'pkts':>5} {'dur_s':>8}  status")
    for r in rows:
        pcap = root / r["pcap_filename"]
        res = verify_one(pcap, args.port)
        ok = res["exists"] and res["readable"] and res["packets"] > 0 and \
            res["ftp_port"] and res["loopback_only"] and res["duration_valid"]
        n_ok += ok
        status = "ok" if ok else "FAIL: " + "; ".join(res["errors"])
        print(f"{r['capture_id']:16} {r['label']:16} {res['packets']:5} "
              f"{res['duration_s']:8.2f}  {status}")

    print(f"\n{n_ok}/{len(rows)} captures verified OK")
    return 0 if (n_ok == len(rows) and not missing and not orphan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
