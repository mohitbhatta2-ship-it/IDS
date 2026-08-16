"""
Collect the MESSY *training* corpus (experimental; production models frozen).

Fresh real captures with ambiguous authentication behaviour and **overlapping
failed-login counts across classes** (benign users who mistype/give up; attackers
who fail or eventually succeed) -- so a model trained on it cannot use raw
failed-login count as a shortcut. Controlled local lab only, NEW addresses/ports,
hash-disjoint from every previous corpus (incl. the frozen 34-PCAP TEST corpus),
labels from the scenario folder.

    python manage.py collect_robust_train_pcaps
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

from predictor import ml, robust_train_capture as rtc, robustness_capture as rc


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / ".git").is_dir():
            return p
    raise RuntimeError("repo root not found")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Command(BaseCommand):
    help = "Collect the messy TRAINING corpus (frozen production models; disjoint from test)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = _repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "robust_train_pcaps"
        for sub in ("benign", "ftp_bruteforce", "metadata"):
            (out / sub).mkdir(parents=True, exist_ok=True)

        targets = rtc.build_targets(out / ".labhome")
        for t in targets.values():
            if not t.is_controlled_local():
                raise SystemExit(f"REFUSING: {t.addr} not controlled-local.")

        w(self.style.MIGRATE_HEADING("Messy TRAINING corpus (real FTP, overlapping failed-login counts)"))
        w(f"  environments: {rtc.train_environments()}  ports: {rtc.TRAIN_PORT}  users: {list(rtc.ALL_USERS)}")
        w(self.style.WARNING("  For TRAINING. Disjoint from the frozen 34-PCAP TEST corpus and all prior corpora."))
        if os.geteuid() != 0:
            w(self.style.WARNING("  note: tcpdump usually needs root."))

        specs = rtc.specs()
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
        pcap = dest / f"{cap_id}_{spec.server}_{spec.scenario}.pcap"
        cap = rc.Tcpdump(pcap_path=pcap, target=target); ts = datetime.now(timezone.utc).isoformat()
        cap.start()
        try:
            stats = spec.fn(target)
        except Exception as e:  # noqa: BLE001 - a client-side error still leaves real captured traffic
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
            verification_status=status,
            detail={**stats, "check_errors": chk.errors})
        (out / "metadata" / f"{cap_id}.json").write_text(json.dumps({**meta.as_row(), "detail": meta.detail}, indent=2, default=str))
        flag = self.style.SUCCESS("ok") if chk.ok else self.style.ERROR("FAIL")
        w(f"  [{flag}] {cap_id:10} {spec.family:16} {spec.scenario:30} pkts={chk.packets:4} dur={chk.duration_s:5.2f}")
        return meta

    def _extract(self, w, metas, out):
        from predictor import pcap_validation as pv, live_capture
        live_capture._ensure_live_on_path()
        w(self.style.MIGRATE_HEADING("\nFeature extraction (existing pipeline, unchanged)"))
        rows, total, bad = [], 0, 0
        for m in metas:
            folder = "benign" if m.label == "Benign" else "ftp_bruteforce"
            pcap = next((out / folder).glob(f"{m.capture_id}_*.pcap"))
            n = ok = inc = 0
            try:
                flows = pv.replay_pcap(pcap)
            except Exception as e:  # noqa: BLE001
                rows.append({"capture_id": m.capture_id, "flows": 0, "valid_flows": 0, "incomplete_flows": 0, "extraction_error": str(e)}); continue
            for fl in flows:
                n += 1
                if pv._feature_problem(fl["features"]):
                    inc += 1
                else:
                    ok += 1
            total += n; bad += inc
            rows.append({"capture_id": m.capture_id, "label": m.label, "flows": n, "valid_flows": ok, "incomplete_flows": inc, "extraction_error": ""})
        pd.DataFrame(rows).to_csv(out / "feature_extraction_report.csv", index=False)
        w(f"  total flows: {total}, incomplete (reported, not zero-filled): {bad}")
        return {"total_flows": int(total), "total_incomplete": int(bad), "n_features": len(ml.FEATURES)}

    def _leakage(self, w, root, out, metas):
        new = {_sha(p) for p in (out).rglob("*.pcap")}
        others = {}
        for name, d in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"),
                        ("independent", "independent_real_pcaps"), ("targeted", "targeted_benign_pcaps"),
                        ("robustness_test", "robustness_pcaps")):
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
        benign = [m for m in metas if m.label == "Benign"]
        bf = [m for m in metas if m.label == "FTP-BruteForce"]

        def dist(key):
            return dict(Counter(getattr(m, key) for m in metas))

        summary = {"experiment": "robust_train_corpus", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "purpose": "Messy real FTP for TRAINING the behavioural candidate (break the failed-login shortcut).",
                   "for_training": True, "labels_from": "scenario folder (never predictions)",
                   "pcap_count": len(metas), "benign_count": len(benign), "ftp_bruteforce_count": len(bf),
                   "captures_passing_verification": sum(m.verification_status == "valid" for m in metas),
                   "total_flows": extraction["total_flows"], "total_incomplete_flows": extraction["total_incomplete"],
                   "diversity": {"by_scenario_family": dist("scenario_family"), "by_client": dist("client"),
                                 "by_server": dist("server"), "by_mode": dist("mode"), "by_environment": dist("environment")},
                   "overlapping_failed_logins": "Both classes contain sessions with 1..N failed logins; benign also succeeds and attackers also eventually succeed.",
                   "leakage": leak,
                   "limitations": ["Loopback / host-local lab only; two server implementations; no FTPS/TLS.",
                                   "Disjoint from the frozen TEST corpus; used for training only in this experiment."],
                   "versions": {"python": platform.python_version()}}
        (out / "collection_report.json").write_text(json.dumps(summary, indent=2, default=str))
        lines = "\n".join(f"| {m.capture_id} | {m.scenario} | {m.scenario_family} | {m.label} | {m.client} | {m.server} | {m.mode} | {m.packet_count} | {m.verification_status} |" for m in metas)
        (out / "collection_report.md").write_text(f"""# Messy TRAINING corpus - collection report

**For TRAINING the behavioural candidate.** Messy real FTP traffic with **overlapping
failed-login counts across both classes** so the model cannot use raw failed-login
count as a label shortcut. Labels from the scenario folder only. Controlled local
lab; NEW addresses (127.0.0.14-16)/ports (2430/2440); hash-disjoint from v1/v2/
independent/targeted **and the frozen 34-PCAP TEST corpus** (leakage all_pass={leak['all_pass']}).

## Totals
- PCAPs: **{len(metas)}** ({len(benign)} benign, {len(bf)} FTP-bruteforce); verified {summary['captures_passing_verification']}/{len(metas)}
- Flows: {extraction['total_flows']} (incomplete: {extraction['total_incomplete']})

## Scenario families
{json.dumps(summary['diversity']['by_scenario_family'], indent=0)}

Benign families include **mistype** (fails then succeeds), **gave_up** (fails then
quits), **multi_user**, **typo_user**, **activity**, **reconnect**, **incomplete**.
Brute-force families include **all_fail** (1/2/4/8 failures) and **eventual_success**
(attacker eventually guesses right). The 1..N failed-login counts overlap the benign
mistype/gave-up sessions on purpose.

## Captures
| id | scenario | family | label | client | server | mode | pkts | verify |
|---|---|---|---|---|---|---|---|---|
{lines}

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
""")
