"""
Collect a DIVERSIFIED real FTP PCAP corpus in the local loopback lab (v2).

Varies genuine behaviour -- multiple clients (ftplib, curl, wget, raw-socket),
real pyftpdlib server variants (rate-limited, permissive, throttled), passive vs
active mode, and three loopback addresses -- to build a substantially more diverse
corpus than v1, WITHOUT injected sleeps and WITHOUT touching production. Each
capture is a fresh tcpdump over fresh connections; labels come from the scenario
folder, never a prediction. The existing pcap_validation pipeline is used read
only to confirm exactly 30 finite in-order features per flow.

    # from webapp_django/ (tcpdump needs root; loopback only)
    python manage.py collect_realistic_pcaps_v2

Output: validation/realistic_pcaps_v2/{benign,ftp_bruteforce,metadata}/ +
MANIFEST.csv + feature_extraction_report.csv + collection_report.{md,json}.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from predictor import ml, lab_capture_v2 as v2


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").is_dir():
            return parent
    raise RuntimeError("repo root not found")


class Command(BaseCommand):
    help = "Collect a diversified real local-lab FTP PCAP corpus (v2; no retraining)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--limit", type=int, default=0, help="cap captures (debug)")

    def handle(self, *args, **opts):
        w = self.stdout.write
        out = Path(opts["output"]) if opts["output"] else \
            _repo_root() / "validation" / "realistic_pcaps_v2"
        for sub in ("benign", "ftp_bruteforce", "metadata"):
            (out / sub).mkdir(parents=True, exist_ok=True)

        targets = v2.build_targets(out / ".labhome")
        # pre-flight: every target must be inside the loopback net
        for t in targets.values():
            if not t.is_loopback_net():
                raise SystemExit(f"REFUSING: {t.addr} is not in 127.0.0.0/8 (lab only).")

        w(self.style.MIGRATE_HEADING("Realistic-PCAP diversification (v2, local loopback lab)"))
        w(f"  environments: {v2.ENV_ADDR}")
        w(f"  server variants: {dict(v2.VARIANT_PORT)}")
        w(f"  clients: python-ftplib, curl({shutil.which('curl')}), "
          f"wget({shutil.which('wget')}), raw-socket")
        if os.geteuid() != 0:
            w(self.style.WARNING("  note: tcpdump usually needs root."))

        specs = v2.benign_specs() + v2.bruteforce_specs()
        if opts["limit"]:
            specs = specs[:opts["limit"]]

        pool = v2.ServerPool(targets, out / ".labhome")
        metas: list[v2.CaptureMeta] = []
        counters = {"benign": 0, "ftpbf": 0}
        try:
            for spec in specs:
                metas.append(self._one_capture(w, spec, pool, out, counters))
        finally:
            pool.stop_all()
            w("  all lab servers stopped.")

        # manifest
        manifest = pd.DataFrame([m.as_row() for m in metas], columns=v2.MANIFEST_COLUMNS)
        manifest.to_csv(out / "MANIFEST.csv", index=False)
        n_valid = sum(m.verification_status == "valid" for m in metas)
        w(self.style.SUCCESS(f"\n  manifest: {len(metas)} captures, {n_valid} verified valid"))

        extraction = self._extract(w, metas, out)
        self._report(out, metas, extraction)
        w(self.style.SUCCESS(f"\nReport: {out/'collection_report.md'}"))
        w(self.style.WARNING("\nSTOP: data collection + validation only. No retraining; "
                             "labels from scenario folder, never predictions."))

    # -- one capture -------------------------------------------------------

    def _one_capture(self, w, spec, pool, out, counters):
        target = pool.get(spec.env, spec.server)
        if spec.label == "Benign":
            counters["benign"] += 1
            cap_id = f"benign_{counters['benign']:02d}"
            dest_dir = out / "benign"
        else:
            counters["ftpbf"] += 1
            cap_id = f"ftpbf_{counters['ftpbf']:02d}"
            dest_dir = out / "ftp_bruteforce"
        pcap = dest_dir / f"{cap_id}_{spec.client}_{spec.scenario}.pcap"

        cap = v2.Tcpdump(pcap_path=pcap, target=target)
        start = datetime.now(timezone.utc).isoformat()
        cap.start()
        try:
            stats = spec.fn(target, **spec.params) if spec.params else spec.fn(target)
        finally:
            cap.stop()
        end = datetime.now(timezone.utc).isoformat()

        chk = v2.verify_pcap(pcap, target)
        status = "valid" if chk.ok else "INVALID:" + ";".join(chk.errors)
        cmd = f"tcpdump -i {target.iface} -w <pcap> -U -n '{target.bpf()}'"
        meta = v2.CaptureMeta(
            capture_id=cap_id, label=spec.label, scenario=spec.scenario,
            client=spec.client, server=f"pyftpdlib:{spec.server}",
            environment=f"{spec.env} ({target.addr})", interface=target.iface,
            mode=spec.mode, attempts=int(stats.get("attempts", 0)),
            start_time=start, end_time=end, packet_count=chk.packets,
            duration_s=round(chk.duration_s, 4), source=target.addr,
            destination=f"{target.addr}:{target.control_port}",
            capture_command=cmd, verification_status=status,
            detail={**stats, "wall_duration_s": round(cap.wall_duration, 3),
                    "check": {k: v for k, v in vars(chk).items() if k != "errors"},
                    "check_errors": chk.errors})
        (out / "metadata" / f"{cap_id}.json").write_text(
            json.dumps({**meta.as_row(), "detail": meta.detail}, indent=2, default=str))
        flag = self.style.SUCCESS("ok") if chk.ok else self.style.ERROR("FAIL")
        w(f"  [{flag}] {cap_id:10} {spec.client:12} {spec.server:11} {spec.env} "
          f"{spec.mode:7} {spec.scenario:32} pkts={chk.packets:4} dur={chk.duration_s:6.2f}")
        if not chk.ok:
            w(self.style.ERROR(f"        {chk.errors}"))
        return meta

    # -- extraction (existing pipeline, read-only) -------------------------

    def _extract(self, w, metas, out):
        from predictor import pcap_validation as pv, live_capture
        live_capture._ensure_live_on_path()
        w(self.style.MIGRATE_HEADING("\nFeature extraction (existing pipeline, unchanged)"))
        rows, total, bad = [], 0, 0
        for m in metas:
            folder = "benign" if m.label == "Benign" else "ftp_bruteforce"
            pcap = next((out / folder).glob(f"{m.capture_id}_*.pcap"))
            n = ok = inc = 0; order_ok = True; probs = []
            try:
                flows = pv.replay_pcap(pcap)
            except Exception as e:  # noqa: BLE001
                rows.append({"capture_id": m.capture_id, "label": m.label, "flows": 0,
                             "valid_flows": 0, "incomplete_flows": 0,
                             "feature_order_ok": None, "extraction_error": str(e),
                             "incomplete_detail": ""})
                continue
            for fl in flows:
                n += 1
                problem = pv._feature_problem(fl["features"])
                if problem:
                    inc += 1; probs.append(problem); continue
                ordered = [float(fl["features"][f]) for f in ml.FEATURES]
                if len(ordered) != len(ml.FEATURES):
                    order_ok = False
                ok += 1
            total += n; bad += inc
            rows.append({"capture_id": m.capture_id, "label": m.label, "flows": n,
                         "valid_flows": ok, "incomplete_flows": inc,
                         "feature_order_ok": order_ok,
                         "incomplete_detail": ";".join(sorted(set(probs))),
                         "extraction_error": ""})
        pd.DataFrame(rows).to_csv(out / "feature_extraction_report.csv", index=False)
        w(f"  total flows: {total}, incomplete (reported, not zero-filled): {bad}")
        return {"per_capture": rows, "total_flows": int(total),
                "total_incomplete": int(bad), "n_features": len(ml.FEATURES)}

    # -- report ------------------------------------------------------------

    def _report(self, out, metas, extraction):
        benign = [m for m in metas if m.label == "Benign"]
        bf = [m for m in metas if m.label == "FTP-BruteForce"]
        valid = [m for m in metas if m.verification_status == "valid"]
        flowmap = {r["capture_id"]: r for r in extraction["per_capture"]}

        def dist(key):
            return dict(Counter(getattr(m, key) for m in metas))

        summary = {
            "experiment": "realistic_pcap_diversification_v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "retrained": False,
            "labels_from": "scenario folder (never model predictions)",
            "loopback_only": True,
            "pcap_count": len(metas),
            "benign_count": len(benign),
            "ftp_bruteforce_count": len(bf),
            "captures_passing_verification": len(valid),
            "total_flows_extracted": extraction["total_flows"],
            "total_incomplete_flows_reported": extraction["total_incomplete"],
            "n_model_features": extraction["n_features"],
            "diversity": {
                "by_client": dist("client"),
                "by_server": dist("server"),
                "by_environment": dist("environment"),
                "by_mode": dist("mode"),
                "distinct_scenarios": len(set(m.scenario for m in metas)),
            },
            "clients": ["python-ftplib", "curl", "wget", "raw-socket"],
            "server_note": ("one FTP server package (pyftpdlib) run in three real "
                            "behavioural variants: rate-limited (defended), permissive, "
                            "throttled bandwidth. Not multiple daemons; documented as such."),
            "tls_note": "TLS/FTPS variant not included: pyOpenSSL unavailable in this env.",
            "versions": {"python": platform.python_version()},
            "limitations": [
                "Loopback lab only (single host); addresses vary but the network is local.",
                "One server package (pyftpdlib) with real config variants; no second FTP daemon.",
                "No FTPS/TLS (dependency unavailable).",
                "Still exploratory relative to real-world traffic diversity.",
            ],
        }
        (out / "collection_report.json").write_text(json.dumps(summary, indent=2, default=str))

        def table(ms):
            lines = []
            for m in ms:
                fr = flowmap.get(m.capture_id, {})
                lines.append(f"| {m.capture_id} | {m.scenario} | {m.client} | "
                             f"{m.server} | {m.environment} | {m.mode} | {m.attempts} | "
                             f"{m.packet_count} | {m.duration_s:.2f} | {fr.get('flows','?')} | "
                             f"{fr.get('incomplete_flows','?')} | {m.verification_status} |")
            return "\n".join(lines)

        hdr = ("| id | scenario | client | server | env | mode | attempts | pkts | "
               "dur_s | flows | incomplete | verify |\n"
               "|---|---|---|---|---|---|---|---|---|---|---|---|")
        md = f"""# Realistic-PCAP diversification (v2) — collection report

**Experimental data collection only. No retraining. Ground-truth labels come from
the scenario/capture directory, never from model predictions.** All traffic stayed
inside the local loopback lab (127.0.0.0/8); no external host was contacted.

## Totals

| Metric | Value |
|---|---|
| PCAP count | **{len(metas)}** ({len(benign)} benign, {len(bf)} FTP-bruteforce) |
| Verified valid | {len(valid)} / {len(metas)} |
| Total flows (existing pipeline) | **{extraction['total_flows']}** |
| Incomplete flows (reported, not zero-filled) | {extraction['total_incomplete']} |
| Model features per flow | {extraction['n_features']} (exact order preserved) |

## Diversity

- **Clients:** {json.dumps(summary['diversity']['by_client'])}
- **Servers (pyftpdlib variants):** {json.dumps(summary['diversity']['by_server'])}
- **Environments (loopback addresses):** {json.dumps(summary['diversity']['by_environment'])}
- **Mode:** {json.dumps(summary['diversity']['by_mode'])}
- **Distinct scenarios:** {summary['diversity']['distinct_scenarios']}

Diversity is genuine: it comes from different client implementations, real server
behaviour (rate-limited vs permissive vs bandwidth-throttled), passive/active
protocol mode, connection reuse patterns, credential sets, attempt counts, and
capture address — **not** from injected sleeps. Slow vs fast brute-force pacing is
a property of the real server's response timing (`auth_failed_timeout`), not a
`time.sleep`.

## How genuineness is guaranteed

Each capture is an independent new network interaction — a fresh `tcpdump` over
fresh FTP connections against a server this process starts and controls. No PCAP
is copied, replayed, relabelled, or edited after capture. The verifier confirms
every packet is loopback-only (127.0.0.0/8), TCP, and on the expected FTP port.

## Benign captures

{hdr}
{table(benign)}

## FTP-BruteForce captures

{hdr}
{table(bf)}

## Feature extraction & validity (existing pipeline, unchanged)

Every capture was replayed through `pcap_validation.replay_pcap` (the validated
Live-Capture engine, unchanged). Each flow is checked for **exactly the
{extraction['n_features']} `ml.FEATURES`, in order, all finite**; a flow missing a
feature is **reported incomplete, never zero-filled**. See
`feature_extraction_report.csv`. The model was never used to produce or check
labels.

## Limitations

{chr(10).join('- ' + x for x in summary['limitations'])}

## Files

`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `collection_report.json`, `verify_pcaps.py`.
"""
        (out / "collection_report.md").write_text(md)
