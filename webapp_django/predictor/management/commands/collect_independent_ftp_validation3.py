"""
Collect the THIRD independent FTP validation corpus (proftpd/ncftp; models frozen; test only).

The FRESH corpus for the per-connection detector: real proftpd server (new), ncftp client,
new netns/subnet (10.99.0.0/24) / port (2424), fresh scenario code with single-session
(packed) vs multi-session attack structure. Hash-disjoint from all training/test corpora.
TEST-ONLY.

    python manage.py collect_independent_ftp_validation3
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from predictor import independent_ftp_lab3 as lab


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / ".git").is_dir():
            return p
    raise RuntimeError("repo root not found")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Command(BaseCommand):
    help = "Collect the 3rd independent FTP corpus (proftpd/ncftp/10.99.0.x; single+multi session; test only)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = _repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "independent_ftp_validation3_pcaps"
        for sub in ("benign", "ftp_bruteforce", "metadata"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        if os.geteuid() != 0:
            raise SystemExit("This collection needs root (netns + proftpd + tcpdump).")
        for tool in ("ip", "proftpd", "ncftp", "tcpdump", "curl"):
            if subprocess.run(["which", tool], capture_output=True).returncode != 0 and not Path(f"/usr/sbin/{tool}").exists():
                raise SystemExit(f"required tool missing: {tool}")

        w(self.style.MIGRATE_HEADING("Third independent FTP corpus (proftpd / ncftp / 10.99.0.x)"))
        w(self.style.WARNING("  TEST-ONLY. Genuinely different stack; hash-disjoint from every prior corpus."))
        the_lab = lab.IvLab3(workdir=out / ".lab").setup()
        w(f"  proftpd up on {lab.SERVER_ADDR}:{lab.PORT}; capturing on {lab.VETH_H} (non-loopback).")

        specs = lab.specs()
        if opts["limit"]:
            specs = specs[:opts["limit"]]
        metas, counters = [], {"benign": 0, "ftpbf": 0}
        try:
            for spec in specs:
                metas.append(self._capture(w, spec, out, counters))
        finally:
            the_lab.stop(); w("  lab stopped (proftpd + netns torn down).")

        manifest = pd.DataFrame([m.as_row() for m in metas], columns=lab.MANIFEST_COLUMNS)
        manifest.to_csv(out / "MANIFEST.csv", index=False)
        n_valid = sum(m.verification_status == "valid" for m in metas)
        w(self.style.SUCCESS(f"\n  manifest: {len(metas)} captures, {n_valid} verified valid"))

        extraction = self._extract(w, metas, out)
        leak = self._leakage(w, root, out, metas)
        self._report(out, metas, extraction, leak)
        w(self.style.SUCCESS(f"\nReport: {out/'collection_report.md'}"))

    def _capture(self, w, spec, out, counters):
        if spec.label == "Benign":
            counters["benign"] += 1; cap_id = f"benign_{counters['benign']:02d}"; dest = out / "benign"
        else:
            counters["ftpbf"] += 1; cap_id = f"ftpbf_{counters['ftpbf']:02d}"; dest = out / "ftp_bruteforce"
        pcap = dest / f"{cap_id}_plain_{spec.scenario}.pcap"
        cap = lab.Capture(pcap_path=pcap); ts = datetime.now(timezone.utc).isoformat()
        cap.start()
        try:
            stats = spec.fn(None)
        except Exception as ex:  # noqa: BLE001
            stats = {"sessions": 0, "client_error": str(ex)}
        finally:
            cap.stop()
        chk = lab.verify_pcap(pcap)
        status = "valid" if chk.ok else "INVALID:" + ";".join(chk.errors)
        meta = lab.CaptureMeta(
            capture_id=cap_id, scenario=spec.scenario, label=spec.label, client=spec.client,
            server="proftpd", environment="netns-ivlab3", interface=lab.VETH_H, mode="passive",
            scenario_family=spec.family, session_structure=spec.structure, encrypted=False,
            sessions=int(stats.get("sessions", 0)), duration_s=round(chk.duration_s, 4), packet_count=chk.packets,
            capture_timestamp=ts, source=lab.HOST_ADDR, destination=f"{lab.SERVER_ADDR}:{lab.PORT}",
            capture_command=f"tcpdump -i {lab.VETH_H} -w <pcap> -U -n '{lab.bpf()}'",
            verification_status=status, detail={**stats, "check_errors": chk.errors})
        (out / "metadata" / f"{cap_id}.json").write_text(json.dumps({**meta.as_row(), "detail": meta.detail}, indent=2, default=str))
        flag = self.style.SUCCESS("ok") if chk.ok else self.style.ERROR("FAIL")
        w(f"  [{flag}] {cap_id:10} {spec.family:16} {spec.scenario:24} {spec.structure:14} sess={stats.get('sessions','?'):>2} pkts={chk.packets:4} dur={chk.duration_s:6.2f}")
        return meta

    def _extract(self, w, metas, out):
        from predictor import pcap_validation as pv, ftp_per_connection as fpc, ftp_cross_session as fcs, live_capture
        live_capture._ensure_live_on_path()
        w(self.style.MIGRATE_HEADING("\nFeature extraction (pipeline + per-connection + cross-session)"))
        rows, total, bad = [], 0, 0
        for m in metas:
            folder = "benign" if m.label == "Benign" else "ftp_bruteforce"
            pcap = next((out / folder).glob(f"{m.capture_id}_*.pcap"))
            pc = fpc.per_connection_features_for_pcap(pcap); cs = fcs.cross_session_features_for_pcap(pcap)
            n = inc = 0
            try:
                flows = pv.replay_pcap(pcap)
            except Exception as e:  # noqa: BLE001
                rows.append({"capture_id": m.capture_id, "flows": 0, "incomplete": 0, "extraction_error": str(e)}); continue
            for fl in flows:
                n += 1
                if pv._feature_problem(fl["features"]):
                    inc += 1
            total += n; bad += inc
            rows.append({"capture_id": m.capture_id, "label": m.label, "session_structure": m.session_structure,
                         "flows": n, "incomplete": inc, "sessions_per_source": cs["ftpx_sessions_per_source"],
                         "max_attempts_per_conn": pc["ftppc_max_attempts_per_conn"],
                         "max_fails_per_conn": pc["ftppc_max_fails_per_conn"]})
        pd.DataFrame(rows).to_csv(out / "feature_extraction_report.csv", index=False)
        w(f"  total flows: {total}, incomplete (reported, not zero-filled): {bad}")
        return {"total_flows": int(total), "total_incomplete": int(bad)}

    def _leakage(self, w, root, out, metas):
        new = {_sha(p) for p in out.rglob("*.pcap")}
        others = {}
        for name, d in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"),
                        ("independent", "independent_real_pcaps"), ("targeted", "targeted_benign_pcaps"),
                        ("robustness", "robustness_pcaps"), ("robust_train", "robust_train_pcaps"),
                        ("benign_failed_login", "benign_failed_login_pcaps"), ("cross_session", "cross_session_pcaps"),
                        ("per_connection", "per_connection_pcaps"),
                        ("independent_ftp_val", "independent_ftp_validation_pcaps"),
                        ("independent_ftp_val2", "independent_ftp_validation2_pcaps")):
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

        summary = {"experiment": "independent_ftp_validation3_corpus", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "purpose": "Fresh independent test for the per-connection + cross-session detector.",
                   "test_only": True, "server": "proftpd (real; not used before)", "client_new": "ncftp",
                   "network": f"veth + ip netns {lab.LAB_NET} (non-loopback, server {lab.SERVER_ADDR})",
                   "pcap_count": len(metas), "benign_count": len(benign), "ftp_bruteforce_count": len(bf),
                   "captures_passing_verification": sum(m.verification_status == "valid" for m in metas),
                   "total_flows": extraction["total_flows"], "total_incomplete_flows": extraction["total_incomplete"],
                   "diversity": {"by_scenario_family": dist("scenario_family"), "by_client": dist("client"),
                                 "by_session_structure": dist("session_structure")},
                   "leakage": leak,
                   "limitations": ["Single container: proftpd + netns give a different server + non-loopback stack, not a "
                                   "separate OS/host.", "Test-only; never training/selection/tuning."],
                   "versions": {"python": platform.python_version(),
                                "proftpd": subprocess.run(["dpkg-query", "-W", "-f=${Version}", "proftpd-basic"], capture_output=True, text=True).stdout.strip()}}
        (out / "collection_report.json").write_text(json.dumps(summary, indent=2, default=str))
        lines = "\n".join(f"| {m.capture_id} | {m.scenario} | {m.scenario_family} | {m.label} | {m.session_structure} | {m.client} | {m.sessions} | {m.packet_count} | {m.verification_status} |" for m in metas)
        (out / "collection_report.md").write_text(f"""# Third independent FTP validation corpus - collection report

**TEST-ONLY. Genuinely different stack: real proftpd server, ncftp client, non-loopback
private network (veth + ip netns {lab.LAB_NET}), fresh scenario code with single-session
(packed) vs multi-session attacks. Labels from folders; hash-disjoint from all prior
corpora and the two earlier independent tests (leakage all_pass={leak['all_pass']}).**

## Totals
- PCAPs: **{len(metas)}** ({len(benign)} benign, {len(bf)} FTP-bruteforce); verified {summary['captures_passing_verification']}/{len(metas)}
- Flows: {extraction['total_flows']} (incomplete: {extraction['total_incomplete']})
- By session structure: {json.dumps(summary['diversity']['by_session_structure'])}

## Captures (sessions = sessions per source)
| id | scenario | family | label | structure | client | sessions | pkts | verify |
|---|---|---|---|---|---|---|---|---|
{lines}

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
""")
