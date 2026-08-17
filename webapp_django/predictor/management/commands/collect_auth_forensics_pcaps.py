"""
Collect the authentication-forensics training corpus (models frozen; TRAIN-only).

Benign typo-then-success / typo-give-up / alternate-password vs single-session dictionary
brute force (incl. dictionary-then-success). Loopback lab, custom raw-socket server, NEW
addresses (127.0.0.50-52) / ports (2830/2840), hash-disjoint from every prior corpus and
every independent test. Labels from the scenario folder.

    python manage.py collect_auth_forensics_pcaps
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from predictor import auth_forensics_capture as afc, robustness_capture as rc, ftp_auth_forensics as faf


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / ".git").is_dir():
            return p
    raise RuntimeError("repo root not found")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Command(BaseCommand):
    help = "Collect the auth-forensics TRAIN corpus (typo-benign vs dictionary-attack)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = _repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "auth_forensics_pcaps"
        for sub in ("benign", "ftp_bruteforce", "metadata"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        targets = afc.build_targets(out / ".labhome")
        for t in targets.values():
            if not t.is_controlled_local():
                raise SystemExit(f"REFUSING: {t.addr} not controlled-local.")
        if os.geteuid() != 0:
            w(self.style.WARNING("  note: tcpdump usually needs root."))

        w(self.style.MIGRATE_HEADING("Auth-forensics corpus (typo-benign vs dictionary-attack)"))
        w(f"  environments: {afc.train_environments()}  ports: {afc.TRAIN_PORT}")
        w(self.style.WARNING("  TRAIN-only. Disjoint from every prior corpus and every independent test."))

        specs = afc.specs()
        if opts["limit"]:
            specs = specs[:opts["limit"]]
        pool = rc.ServerPool(targets, out / ".labhome")
        metas, counters = [], {"benign": 0, "ftpbf": 0}
        try:
            for spec in specs:
                metas.append(self._capture(w, spec, pool, out, counters))
        finally:
            pool.stop_all(); w("  servers stopped.")

        manifest = pd.DataFrame([m.as_row() for m in metas], columns=rc.MANIFEST_COLUMNS)
        manifest.to_csv(out / "MANIFEST.csv", index=False)
        n_valid = sum(m.verification_status == "valid" for m in metas)
        w(self.style.SUCCESS(f"\n  manifest: {len(metas)} captures, {n_valid} verified valid"))

        extraction = self._extract(w, metas, out)
        leak = self._leakage(w, root, out, metas)
        self._report(out, metas, extraction, leak)
        w(self.style.SUCCESS(f"\nReport: {out/'collection_report.md'}"))

    def _capture(self, w, spec, pool, out, counters):
        target = pool.get(spec.env, spec.server)
        if spec.label == "Benign":
            counters["benign"] += 1; cap_id = f"benign_{counters['benign']:02d}"; dest = out / "benign"
        else:
            counters["ftpbf"] += 1; cap_id = f"ftpbf_{counters['ftpbf']:02d}"; dest = out / "ftp_bruteforce"
        pcap = dest / f"{cap_id}_custom_{spec.scenario}.pcap"
        cap = rc.Tcpdump(pcap_path=pcap, target=target); ts = datetime.now(timezone.utc).isoformat()
        cap.start()
        try:
            stats = spec.fn(target)
        except Exception as ex:  # noqa: BLE001
            stats = {"sessions": 0, "client_error": str(ex)}
        finally:
            cap.stop()
        chk = rc.verify_pcap(pcap, target)
        status = "valid" if chk.ok else "INVALID:" + ";".join(chk.errors)
        meta = rc.CaptureMeta(
            capture_id=cap_id, scenario=spec.scenario, label=spec.label, client=spec.client,
            server="custom(raw-socket)", environment=f"{spec.env} ({target.addr})", interface=target.iface,
            mode=spec.mode, scenario_family=spec.family, attempts=int(stats.get("sessions", 0)),
            duration_s=round(chk.duration_s, 4), packet_count=chk.packets, capture_timestamp=ts,
            source=target.addr, destination=f"{target.addr}:{target.control_port}",
            capture_command=f"tcpdump -i {target.iface} -w <pcap> -U -n '{target.bpf()}'",
            verification_status=status, detail={**stats, "check_errors": chk.errors})
        (out / "metadata" / f"{cap_id}.json").write_text(json.dumps({**meta.as_row(), "detail": meta.detail}, indent=2, default=str))
        flag = self.style.SUCCESS("ok") if chk.ok else self.style.ERROR("FAIL")
        w(f"  [{flag}] {cap_id:10} {spec.family:14} {spec.scenario:24} pkts={chk.packets:4} dur={chk.duration_s:6.2f}")
        return meta

    def _extract(self, w, metas, out):
        from predictor import pcap_validation as pv, live_capture
        live_capture._ensure_live_on_path()
        w(self.style.MIGRATE_HEADING("\nFeature extraction (pipeline + auth-forensics)"))
        rows, total, bad = [], 0, 0
        for m in metas:
            folder = "benign" if m.label == "Benign" else "ftp_bruteforce"
            pcap = next((out / folder).glob(f"{m.capture_id}_*.pcap"))
            af = faf.forensic_features_for_pcap(pcap)
            n = inc = 0
            for fl in pv.replay_pcap(pcap):
                n += 1
                if pv._feature_problem(fl["features"]):
                    inc += 1
            total += n; bad += inc
            rows.append({"capture_id": m.capture_id, "label": m.label, "family": m.scenario_family, "flows": n, "incomplete": inc,
                         "distinct_failed_pw": af["ftpaf_distinct_failed_pw"],
                         "min_editdist_fail_to_success": af["ftpaf_min_editdist_fail_to_success"],
                         "mean_editdist_consecutive": af["ftpaf_mean_editdist_consecutive"],
                         "fails_before_success": af["ftpaf_fails_before_success"]})
        pd.DataFrame(rows).to_csv(out / "feature_extraction_report.csv", index=False)
        w(f"  total flows: {total}, incomplete: {bad}")
        return {"total_flows": int(total), "total_incomplete": int(bad)}

    def _leakage(self, w, root, out, metas):
        new = {_sha(p) for p in out.rglob("*.pcap")}
        others = {}
        for name, d in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"), ("independent", "independent_real_pcaps"),
                        ("targeted", "targeted_benign_pcaps"), ("robustness", "robustness_pcaps"),
                        ("robust_train", "robust_train_pcaps"), ("benign_failed_login", "benign_failed_login_pcaps"),
                        ("cross_session", "cross_session_pcaps"), ("per_connection", "per_connection_pcaps"),
                        ("independent_ftp_val", "independent_ftp_validation_pcaps"),
                        ("independent_ftp_val2", "independent_ftp_validation2_pcaps"),
                        ("independent_ftp_val3", "independent_ftp_validation3_pcaps")):
            others[name] = {_sha(p) for p in (root / "validation" / d).rglob("*.pcap")}
        checks = {f"disjoint_from_{n}": new.isdisjoint(h) for n, h in others.items()}
        checks["no_duplicate_within_corpus"] = len(new) == len(list(out.rglob("*.pcap")))
        checks["labels_only_from_folders"] = all(m.label in ("Benign", "FTP-BruteForce") for m in metas)
        checks["all_verified_valid"] = all(m.verification_status == "valid" for m in metas)
        res = {"all_pass": all(checks.values()), "checks": checks, "n_new": len(new)}
        (out / "leakage_validation.json").write_text(json.dumps(res, indent=2, default=str))
        w(self.style.MIGRATE_HEADING("\nLeakage / disjointness"))
        w(f"  all_pass={res['all_pass']}  " + " ".join(f"{k}={v}" for k, v in checks.items()))
        return res

    def _report(self, out, metas, extraction, leak):
        benign = [m for m in metas if m.label == "Benign"]; bf = [m for m in metas if m.label == "FTP-BruteForce"]

        def dist(key):
            return dict(Counter(getattr(m, key) for m in metas))

        summary = {"experiment": "auth_forensics_corpus", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "purpose": "Typo-benign vs dictionary-attack for the mistype-vs-single-session-attack distinction.",
                   "for_training": True, "labels_from": "scenario folder", "pcap_count": len(metas),
                   "benign_count": len(benign), "ftp_bruteforce_count": len(bf),
                   "captures_passing_verification": sum(m.verification_status == "valid" for m in metas),
                   "total_flows": extraction["total_flows"], "total_incomplete_flows": extraction["total_incomplete"],
                   "diversity": {"by_scenario_family": dist("scenario_family")}, "leakage": leak,
                   "limitations": ["Loopback lab; custom raw-socket server; benign typos are synthetic single-edit typos.",
                                   "TRAIN-only; disjoint from every independent test."],
                   "versions": {"python": platform.python_version()}}
        (out / "collection_report.json").write_text(json.dumps(summary, indent=2, default=str))
        lines = "\n".join(f"| {m.capture_id} | {m.scenario} | {m.scenario_family} | {m.label} | {m.packet_count} | {m.verification_status} |" for m in metas)
        (out / "collection_report.md").write_text(f"""# Auth-forensics training corpus - collection report

**TRAIN-only. Benign typo-then-success / typo-give-up / alternate-password vs single-session
dictionary brute force (incl. dictionary-then-success). Loopback lab; NEW addresses
(127.0.0.50-52) / ports (2830/2840); hash-disjoint from all prior corpora and every
independent test (leakage all_pass={leak['all_pass']}).**

## Totals
- PCAPs: **{len(metas)}** ({len(benign)} benign, {len(bf)} FTP-bruteforce); verified {summary['captures_passing_verification']}/{len(metas)}
- Flows: {extraction['total_flows']} (incomplete: {extraction['total_incomplete']})
- Families: {json.dumps(summary['diversity']['by_scenario_family'])}

## Captures
| id | scenario | family | label | pkts | verify |
|---|---|---|---|---|---|
{lines}

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
""")
