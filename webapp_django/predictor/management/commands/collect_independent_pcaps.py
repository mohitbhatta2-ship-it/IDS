"""
Collect the INDEPENDENT real-PCAP test corpus (experimental; models frozen).

Captures a completely NEW real FTP corpus for the final independent evaluation of
the frozen Candidate 2 -- a genuinely different server implementation (raw-socket
custom FTP server) alongside pyftpdlib, new command sequences, new addresses, and
new ports. These captures are for TESTING ONLY: never used for training, tuning,
weighting, threshold/candidate selection, or SHAP tuning. Ground truth is the
scenario folder; the model is never consulted.

    # from webapp_django/ (tcpdump needs root; controlled local lab only)
    python manage.py collect_independent_pcaps

Output: validation/independent_real_pcaps/{benign,ftp_bruteforce,metadata}/ +
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

from predictor import ml, independent_capture as ic


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / ".git").is_dir():
            return p
    raise RuntimeError("repo root not found")


class Command(BaseCommand):
    help = "Collect the independent real-PCAP TEST corpus (frozen models; no training)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **opts):
        w = self.stdout.write
        out = Path(opts["output"]) if opts["output"] else \
            _repo_root() / "validation" / "independent_real_pcaps"
        for sub in ("benign", "ftp_bruteforce", "metadata"):
            (out / sub).mkdir(parents=True, exist_ok=True)

        targets = ic.build_targets(out / ".labhome")
        for t in targets.values():
            if not t.is_controlled_local():
                raise SystemExit(f"REFUSING: {t.addr} is not a controlled local address.")

        w(self.style.MIGRATE_HEADING("Independent real-PCAP TEST corpus collection"))
        w(f"  environments: {ic.environments()}")
        w(f"  servers: custom(raw-socket), pyftpdlib(permissive/ratelimited)  "
          f"ports={ic.SERVER_PORT}")
        w(f"  clients: python-ftplib, curl({shutil.which('curl')}), "
          f"wget({shutil.which('wget')}), raw-socket")
        w(self.style.WARNING("  TEST-ONLY corpus: never used for training/tuning/selection."))
        if os.geteuid() != 0:
            w(self.style.WARNING("  note: tcpdump usually needs root."))

        specs = ic.benign_specs() + ic.bruteforce_specs()
        if opts["limit"]:
            specs = specs[:opts["limit"]]

        pool = ic.ServerPool(targets, out / ".labhome")
        metas, counters = [], {"benign": 0, "ftpbf": 0}
        try:
            for spec in specs:
                metas.append(self._capture(w, spec, pool, out, counters))
        finally:
            pool.stop_all()
            w("  all lab servers stopped.")

        manifest = pd.DataFrame([m.as_row() for m in metas], columns=ic.MANIFEST_COLUMNS)
        manifest.to_csv(out / "MANIFEST.csv", index=False)
        n_valid = sum(m.verification_status == "valid" for m in metas)
        w(self.style.SUCCESS(f"\n  manifest: {len(metas)} captures, {n_valid} verified valid"))

        extraction = self._extract(w, metas, out)
        self._report(out, metas, extraction)
        w(self.style.SUCCESS(f"\nReport: {out/'collection_report.md'}"))
        w(self.style.WARNING("\nSTOP: TEST corpus collected. Do not use for training/tuning."))

    def _capture(self, w, spec, pool, out, counters):
        target = pool.get(spec.env, spec.server)
        if spec.label == "Benign":
            counters["benign"] += 1
            cap_id = f"benign_{counters['benign']:02d}"; dest = out / "benign"
        else:
            counters["ftpbf"] += 1
            cap_id = f"ftpbf_{counters['ftpbf']:02d}"; dest = out / "ftp_bruteforce"
        pcap = dest / f"{cap_id}_{spec.server}_{spec.scenario}.pcap"

        cap = ic.Tcpdump(pcap_path=pcap, target=target)
        ts = datetime.now(timezone.utc).isoformat()
        cap.start()
        try:
            stats = spec.fn(target)
        finally:
            cap.stop()

        chk = ic.verify_pcap(pcap, target)
        status = "valid" if chk.ok else "INVALID:" + ";".join(chk.errors)
        cmd = f"tcpdump -i {target.iface} -w <pcap> -U -n '{target.bpf()}'"
        meta = ic.CaptureMeta(
            capture_id=cap_id, scenario=spec.scenario, label=spec.label, client=spec.client,
            server=f"{spec.server}" + ("(custom-raw-socket)" if spec.server == "custom" else "(pyftpdlib)"),
            environment=f"{spec.env} ({target.addr})", interface=target.iface,
            connection_pattern=spec.pattern, attempts=int(stats.get("attempts", 0)),
            duration_s=round(chk.duration_s, 4), packet_count=chk.packets,
            capture_timestamp=ts, source=target.addr,
            destination=f"{target.addr}:{target.control_port}", capture_command=cmd,
            verification_status=status,
            detail={**stats, "mode": spec.mode, "wall_duration_s": round(cap.wall_duration, 3),
                    "check": {k: v for k, v in vars(chk).items() if k != "errors"},
                    "check_errors": chk.errors})
        (out / "metadata" / f"{cap_id}.json").write_text(
            json.dumps({**meta.as_row(), "detail": meta.detail}, indent=2, default=str))
        flag = self.style.SUCCESS("ok") if chk.ok else self.style.ERROR("FAIL")
        w(f"  [{flag}] {cap_id:10} {spec.client:12} {spec.server:11} {spec.env:6} "
          f"{spec.scenario:28} pkts={chk.packets:4} dur={chk.duration_s:6.2f}")
        if not chk.ok:
            w(self.style.ERROR(f"        {chk.errors}"))
        return meta

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
                             "valid_flows": 0, "incomplete_flows": 0, "feature_order_ok": None,
                             "incomplete_detail": "", "extraction_error": str(e)})
                continue
            for fl in flows:
                n += 1
                problem = pv._feature_problem(fl["features"])
                if problem:
                    inc += 1; probs.append(problem); continue
                if len([f for f in ml.FEATURES if f in fl["features"]]) != 30:
                    order_ok = False
                ok += 1
            total += n; bad += inc
            rows.append({"capture_id": m.capture_id, "label": m.label, "flows": n,
                         "valid_flows": ok, "incomplete_flows": inc, "feature_order_ok": order_ok,
                         "incomplete_detail": ";".join(sorted(set(probs))), "extraction_error": ""})
            w(f"  {m.capture_id:10} {m.label:14} flows={n:3} valid={ok:3} incomplete={inc}")
        pd.DataFrame(rows).to_csv(out / "feature_extraction_report.csv", index=False)
        w(f"  total flows: {total}, incomplete (reported, not zero-filled): {bad}")
        return {"per_capture": rows, "total_flows": int(total), "total_incomplete": int(bad),
                "n_features": len(ml.FEATURES)}

    def _report(self, out, metas, extraction):
        benign = [m for m in metas if m.label == "Benign"]
        bf = [m for m in metas if m.label == "FTP-BruteForce"]
        valid = [m for m in metas if m.verification_status == "valid"]

        def dist(key):
            return dict(Counter(getattr(m, key) for m in metas))

        summary = {
            "experiment": "independent_real_pcap_test_corpus",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "TEST ONLY -- final independent evaluation of frozen Candidate 2.",
            "never_used_for": ["training", "retraining", "fine-tuning", "sample weighting",
                               "threshold selection", "hyperparameter selection",
                               "feature selection", "candidate selection", "SHAP tuning"],
            "labels_from": "scenario folder (never model predictions)",
            "controlled_local_only": True,
            "pcap_count": len(metas), "benign_count": len(benign),
            "ftp_bruteforce_count": len(bf), "captures_passing_verification": len(valid),
            "total_flows": extraction["total_flows"],
            "total_incomplete_flows": extraction["total_incomplete"],
            "n_model_features": extraction["n_features"],
            "diversity": {"by_client": dist("client"), "by_server": dist("server"),
                          "by_environment": dist("environment"),
                          "by_connection_pattern": dist("connection_pattern"),
                          "distinct_scenarios": len(set(m.scenario for m in metas))},
            "new_vs_v2": ["custom raw-socket FTP server (different implementation)",
                          "FTP commands not in v2 (MKD/RMD, RNFR/RNTO, DELE, APPE, SIZE, MDTM, NLST, STAT)",
                          "new loopback addresses (127.0.0.5/.6) + host-local address",
                          "new control ports (2130/2140/2150)"],
            "limitations": [
                "Single-host container: same-host traffic always traverses lo, so no "
                "genuinely separate physical interface is exercised (address varies, not iface).",
                "Two server implementations (custom raw-socket + pyftpdlib); no third-party daemon.",
                "No FTPS/TLS (pyOpenSSL unavailable).",
                "Loopback lab; not real multi-host network traffic.",
            ],
            "versions": {"python": platform.python_version()},
        }
        (out / "collection_report.json").write_text(json.dumps(summary, indent=2, default=str))

        flowmap = {r["capture_id"]: r for r in extraction["per_capture"]}

        def table(ms):
            out_lines = []
            for m in ms:
                fr = flowmap.get(m.capture_id, {})
                out_lines.append(f"| {m.capture_id} | {m.scenario} | {m.client} | {m.server} | "
                                 f"{m.environment} | {m.connection_pattern} | {m.attempts} | "
                                 f"{m.packet_count} | {m.duration_s:.2f} | {fr.get('flows','?')} | "
                                 f"{fr.get('incomplete_flows','?')} | {m.verification_status} |")
            return "\n".join(out_lines)

        hdr = ("| id | scenario | client | server | env | pattern | attempts | pkts | dur_s | "
               "flows | incomplete | verify |\n|" + "---|" * 12)
        md = f"""# Independent real-PCAP TEST corpus — collection report

**TEST-ONLY corpus for the final independent evaluation of the frozen Candidate 2.
These PCAPs were NEVER used for training, retraining, tuning, sample weighting,
threshold selection, hyperparameter/feature/candidate selection, or SHAP tuning.**
Ground-truth labels come only from the scenario/capture directory. All traffic
stayed inside the controlled local lab; no external host was contacted.

## Totals

| Metric | Value |
|---|---|
| PCAP count | **{len(metas)}** ({len(benign)} benign, {len(bf)} FTP-bruteforce) |
| Verified valid | {len(valid)} / {len(metas)} |
| Total flows (existing pipeline) | **{extraction['total_flows']}** |
| Incomplete flows (reported, not zero-filled) | {extraction['total_incomplete']} |
| Model features per flow | {extraction['n_features']} (exact order) |

## Diversity (genuine — new relative to v2)

- **Clients:** {json.dumps(summary['diversity']['by_client'])}
- **Servers:** {json.dumps(summary['diversity']['by_server'])} — includes a
  hand-written raw-socket FTP server, a genuinely different implementation.
- **Environments:** {json.dumps(summary['diversity']['by_environment'])}
- **Connection patterns:** {json.dumps(summary['diversity']['by_connection_pattern'])}
- **Distinct scenarios:** {summary['diversity']['distinct_scenarios']}
- **New command sequences vs v2:** MKD/RMD, RNFR/RNTO, DELE, APPE, SIZE, MDTM,
  NLST, STAT, active-mode data, upload-then-download.

## Benign captures

{hdr}
{table(benign)}

## FTP-BruteForce captures

{hdr}
{table(bf)}

## Honest limitations

{chr(10).join('- ' + x for x in summary['limitations'])}

## Files

`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `collection_report.json`, `verify_pcaps.py`.
"""
        (out / "collection_report.md").write_text(md)
