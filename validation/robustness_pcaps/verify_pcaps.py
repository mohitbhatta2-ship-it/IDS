#!/usr/bin/env python3
"""
Standalone verifier for the INDEPENDENT real-PCAP test corpus (no Django, no model).

Reads MANIFEST.csv and independently re-verifies every referenced PCAP with scapy:
exists, readable, packets present, FTP traffic on the expected control port,
controlled-local traffic only (127.0.0.0/8 or the host's own eth0 IP -- never
external), TCP only, finite duration. Also checks manifest<->disk agreement and
that there are no duplicate PCAPs (by content hash). Exits non-zero on any failure.

Labels are read from the manifest (scenario/folder), never inferred from a model.

    python3 validation/independent_real_pcaps/verify_pcaps.py [dir]
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import ipaddress
import math
import socket
import struct
import sys
from pathlib import Path

LOOPBACK = ipaddress.ip_network("127.0.0.0/8")


def _host_eth0():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915,
                                            struct.pack("256s", b"eth0"))[20:24])
    except Exception:  # noqa: BLE001
        return None


def _controlled(addr, eth0):
    return ipaddress.ip_address(addr) in LOOPBACK or (eth0 and addr == eth0)


def verify_one(pcap: Path, host: str, port: int, eth0) -> dict:
    from scapy.all import rdpcap, TCP, UDP, IP
    r = {"pcap": pcap.name, "exists": pcap.is_file(), "readable": False, "packets": 0,
         "controlled_local": True, "host_present": False, "port_ok": False, "tcp_only": True,
         "duration_s": 0.0, "duration_valid": False, "errors": []}
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
                if not _controlled(a, eth0):
                    r["controlled_local"] = False
            if host in (pk[IP].src, pk[IP].dst):
                r["host_present"] = True
        if TCP in pk:
            if port in (pk[TCP].sport, pk[TCP].dport):
                r["port_ok"] = True
        elif UDP in pk:
            r["tcp_only"] = False
    if times:
        d = max(times) - min(times)
        r["duration_s"] = round(d, 4); r["duration_valid"] = d >= 0 and math.isfinite(d)
    if not r["controlled_local"]:
        r["errors"].append("non-controlled (external) address")
    if not r["port_ok"]:
        r["errors"].append(f"no traffic on port {port}")
    if not r["tcp_only"]:
        r["errors"].append("non-TCP payload")
    if not r["duration_valid"]:
        r["errors"].append("invalid duration")
    return r


def main(argv):
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent
    manifest = root / "MANIFEST.csv"
    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr); return 2
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("ERROR: empty manifest", file=sys.stderr); return 2
    eth0 = _host_eth0()

    hashes, dups = {}, []
    n_ok, missing = 0, []
    print(f"Verifying {len(rows)} captures under {root} (host eth0={eth0})\n")
    print(f"{'capture':12} {'label':16} {'host':11} {'port':>5} {'pkts':>5} {'dur_s':>7}  status")
    for r in rows:
        folder = "benign" if r["label"] == "Benign" else "ftp_bruteforce"
        matches = list((root / folder).glob(f"{r['capture_id']}_*.pcap"))
        if not matches:
            missing.append(r["capture_id"])
            print(f"{r['capture_id']:12} {r['label']:16} MISSING"); continue
        pcap = matches[0]
        h = hashlib.sha256(pcap.read_bytes()).hexdigest()
        if h in hashes:
            dups.append((pcap.name, hashes[h]))
        hashes[h] = pcap.name
        host = r["source"]; port = int(r["destination"].rsplit(":", 1)[1])
        res = verify_one(pcap, host, port, eth0)
        ok = (res["exists"] and res["readable"] and res["packets"] > 0 and res["controlled_local"]
              and res["host_present"] and res["port_ok"] and res["tcp_only"] and res["duration_valid"])
        n_ok += ok
        status = "ok" if ok else "FAIL: " + "; ".join(res["errors"])
        print(f"{r['capture_id']:12} {r['label']:16} {host:11} {port:5} {res['packets']:5} "
              f"{res['duration_s']:7.2f}  {status}")
    if dups:
        print(f"\nERROR: duplicate PCAPs: {dups}", file=sys.stderr)
    print(f"\n{n_ok}/{len(rows)} captures verified OK; duplicates={len(dups)}; missing={len(missing)}")
    return 0 if (n_ok == len(rows) and not dups and not missing) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
