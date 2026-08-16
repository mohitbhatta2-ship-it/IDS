"""
Collect the realistic benign FAILED-LOGIN training corpus (models frozen; TRAIN-only).

Fresh benign FTP traffic with the *dynamics* that separate a human who mis-typed from a
brute force: think-time between attempts, password REUSE, small attempt counts, graceful
QUIT, and real activity after an eventual success. Loopback lab, NEW addresses
(127.0.0.20-22) / ports (2530/2540), hash-disjoint from every prior corpus AND the
frozen independent vsFTPD test. Labels from the scenario folder.

    python manage.py collect_benign_failed_login_pcaps
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

from predictor import ml, benign_failed_login_capture as bfl, robustness_capture as rc, \
    ftp_behavioral as fb, ftp_behavioral_ext as fbx


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / ".git").is_dir():
            return p
    raise RuntimeError("repo root not found")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Command(BaseCommand):
    help = "Collect the realistic benign failed-login TRAIN corpus (frozen models; disjoint from test)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = _repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "benign_failed_login_pcaps"
        for sub in ("benign", "ftp_bruteforce", "metadata"):
            (out / sub).mkdir(parents=True, exist_ok=True)

        targets = bfl.build_targets(out / ".labhome")
        for t in targets.values():
            if not t.is_controlled_local():
                raise SystemExit(f"REFUSING: {t.addr} not controlled-local.")
        if os.geteuid() != 0:
            w(self.style.WARNING("  note: tcpdump usually needs root."))

        w(self.style.MIGRATE_HEADING("Realistic benign failed-login corpus (human pacing / reuse / post-auth activity)"))
        w(f"  environments: {bfl.train_environments()}  ports: {bfl.TRAIN_PORT}")
        w(self.style.WARNING("  TRAIN-only. Disjoint from the frozen independent vsFTPD test and all prior corpora."))

        specs = bfl.specs()
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
        counters["benign"] += 1; cap_id = f"benign_{counters['benign']:02d}"; dest = out / "benign"
        pcap = dest / f"{cap_id}_{spec.server}_{spec.scenario}.pcap"
        cap = rc.Tcpdump(pcap_path=pcap, target=target); ts = datetime.now(timezone.utc).isoformat()
        cap.start()
        try:
            stats = spec.fn(target)
        except Exception as e:  # noqa: BLE001
            stats = {"attempts": 0, "client_error": str(e)}
        finally:
            cap.stop()
        chk = rc.verify_pcap(pcap, target)
        status = "valid" if chk.ok else "INVALID:" + ";".join(chk.errors)
        meta = rc.CaptureMeta(
            capture_id=cap_id, scenario=spec.scenario, label=spec.label, client=spec.client,
            server=f"{spec.server}" + ("(custom-raw-socket)" if spec.server == "custom" else "(pyftpdlib)"),
            environment=f"{spec.env} ({target.addr})", interface=target.iface, mode=spec.mode,
            scenario_family=spec.family, attempts=int(stats.get("attempts", 0)),
            duration_s=round(chk.duration_s, 4), packet_count=chk.packets, capture_timestamp=ts,
            source=target.addr, destination=f"{target.addr}:{target.control_port}",
            capture_command=f"tcpdump -i {target.iface} -w <pcap> -U -n '{target.bpf()}'",
            verification_status=status, detail={**stats, "check_errors": chk.errors})
        (out / "metadata" / f"{cap_id}.json").write_text(json.dumps({**meta.as_row(), "detail": meta.detail}, indent=2, default=str))
        flag = self.style.SUCCESS("ok") if chk.ok else self.style.ERROR("FAIL")
        w(f"  [{flag}] {cap_id:10} {spec.scenario:26} pkts={chk.packets:4} dur={chk.duration_s:6.2f}")
        return meta

    def _extract(self, w, metas, out):
        from predictor import pcap_validation as pv, live_capture
        live_capture._ensure_live_on_path()
        w(self.style.MIGRATE_HEADING("\nFeature extraction (existing pipeline + ext features)"))
        rows, total, bad = [], 0, 0
        for m in metas:
            pcap = next((out / "benign").glob(f"{m.capture_id}_*.pcap"))
            behav = fb.behavioural_features_for_pcap(pcap); ext = fbx.ext_features_for_pcap(pcap)
            n = ok = inc = 0
            for fl in pv.replay_pcap(pcap):
                n += 1
                if pv._feature_problem(fl["features"]):
                    inc += 1
                else:
                    ok += 1
            total += n; bad += inc
            rows.append({"capture_id": m.capture_id, "scenario": m.scenario, "flows": n, "incomplete": inc,
                         "failed_logins": behav["ftp_failed_logins"], "has_auth": behav["ftp_has_successful_auth"],
                         "distinct_pw": ext["ftpx_distinct_passwords"], "reuse_ratio": ext["ftpx_password_reuse_ratio"],
                         "interattempt_mean_s": ext["ftpx_interattempt_mean_s"],
                         "post_auth_cmds": ext["ftpx_post_auth_commands"], "ended_with_quit": ext["ftpx_ended_with_quit"]})
        pd.DataFrame(rows).to_csv(out / "feature_extraction_report.csv", index=False)
        w(f"  total flows: {total}, incomplete: {bad}")
        return {"total_flows": int(total), "total_incomplete": int(bad)}

    def _leakage(self, w, root, out, metas):
        new = {_sha(p) for p in out.rglob("*.pcap")}
        others = {}
        for name, d in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"),
                        ("independent", "independent_real_pcaps"), ("targeted", "targeted_benign_pcaps"),
                        ("robustness", "robustness_pcaps"), ("robust_train", "robust_train_pcaps"),
                        ("independent_ftp_val", "independent_ftp_validation_pcaps")):
            others[name] = {_sha(p) for p in (root / "validation" / d).rglob("*.pcap")}
        checks = {f"disjoint_from_{n}": new.isdisjoint(h) for n, h in others.items()}
        checks["no_duplicate_within_corpus"] = len(new) == len(list(out.rglob("*.pcap")))
        checks["labels_only_from_folders"] = all(m.label == "Benign" for m in metas)
        checks["all_verified_valid"] = all(m.verification_status == "valid" for m in metas)
        checks["disjoint_from_independent_ftp_TEST"] = checks["disjoint_from_independent_ftp_val"]
        res = {"all_pass": all(checks.values()), "checks": checks, "n_new": len(new)}
        (out / "leakage_validation.json").write_text(json.dumps(res, indent=2, default=str))
        w(self.style.MIGRATE_HEADING("\nLeakage / disjointness"))
        w(f"  all_pass={res['all_pass']}  " + " ".join(f"{k}={v}" for k, v in checks.items()))
        return res

    def _report(self, out, metas, extraction, leak):
        def dist(key):
            return dict(Counter(getattr(m, key) for m in metas))

        summary = {"experiment": "benign_failed_login_corpus", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "purpose": "Realistic benign failed-login dynamics (pacing/reuse/post-auth activity) for TRAINING.",
                   "for_training": True, "labels_from": "scenario folder", "pcap_count": len(metas),
                   "captures_passing_verification": sum(m.verification_status == "valid" for m in metas),
                   "total_flows": extraction["total_flows"], "total_incomplete_flows": extraction["total_incomplete"],
                   "diversity": {"by_scenario_family": dist("scenario_family"), "by_server": dist("server"),
                                 "by_mode": dist("mode"), "by_environment": dist("environment")},
                   "leakage": leak,
                   "limitations": ["Loopback lab; pyftpdlib + custom raw-socket servers (NOT vsftpd, to keep the "
                                   "vsFTPD test held-out); benign-only; think-times are synthetic-but-real timing.",
                                   "TRAIN-only; disjoint from the frozen independent vsFTPD test."],
                   "versions": {"python": platform.python_version()}}
        (out / "collection_report.json").write_text(json.dumps(summary, indent=2, default=str))
        lines = "\n".join(f"| {m.capture_id} | {m.scenario} | {m.scenario_family} | {m.server} | {m.mode} | {m.packet_count} | {m.duration_s} | {m.verification_status} |" for m in metas)
        (out / "collection_report.md").write_text(f"""# Realistic benign failed-login corpus - collection report

**TRAIN-only. Benign FTP traffic with realistic *dynamics* (think-time pacing, password
REUSE, small attempt counts, graceful QUIT, post-success activity) to teach the model
"human who mis-typed" vs "brute force". Loopback lab; NEW addresses (127.0.0.20-22) /
ports (2530/2540); hash-disjoint from all prior corpora and the frozen independent
vsFTPD test (leakage all_pass={leak['all_pass']}).**

## Totals
- PCAPs: **{len(metas)}** (all Benign); verified {summary['captures_passing_verification']}/{len(metas)}
- Flows: {extraction['total_flows']} (incomplete: {extraction['total_incomplete']})
- Families: {json.dumps(summary['diversity']['by_scenario_family'])}

## Captures
| id | scenario | family | server | mode | pkts | dur(s) | verify |
|---|---|---|---|---|---|---|---|
{lines}

## Files
`MANIFEST.csv`, `benign/*.pcap`, `metadata/*.json`, `feature_extraction_report.csv`,
`leakage_validation.json`, `collection_report.json`.
""")
