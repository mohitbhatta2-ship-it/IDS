"""
Collect the INDEPENDENT FTP validation corpus (experimental; models frozen; test only).

Genuinely different traffic from every prior corpus: real ``vsftpd`` server, ``lftp``
client, non-loopback private network (veth + ip netns, 10.77.0.0/24), plus separate
FTPS/TLS captures. Fresh scenario code (``independent_ftp_lab``); labels from the
scenario folder; hash-disjoint from all training/test corpora. TEST-ONLY.

    python manage.py collect_independent_ftp_validation
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

from predictor import ml, independent_ftp_lab as lab


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / ".git").is_dir():
            return p
    raise RuntimeError("repo root not found")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Command(BaseCommand):
    help = "Collect the independent FTP validation corpus (real vsftpd/lftp, non-loopback; test only)."

    def add_arguments(self, parser):
        parser.add_argument("--output", default=None)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **opts):
        w = self.stdout.write
        root = _repo_root()
        out = Path(opts["output"]) if opts["output"] else root / "validation" / "independent_ftp_validation_pcaps"
        for sub in ("benign", "ftp_bruteforce", "metadata"):
            (out / sub).mkdir(parents=True, exist_ok=True)

        if os.geteuid() != 0:
            raise SystemExit("This collection needs root (netns + vsftpd + tcpdump).")
        for tool in ("ip", "vsftpd", "lftp", "tcpdump", "curl"):
            if subprocess.run(["which", tool], capture_output=True).returncode != 0 and \
               not Path(f"/usr/sbin/{tool}").exists():
                raise SystemExit(f"required tool missing: {tool}")

        w(self.style.MIGRATE_HEADING("Independent FTP validation corpus (real vsftpd / lftp / non-loopback)"))
        w(f"  server: vsFTPd  network: veth+netns {lab.LAB_NET} (server {lab.SERVER_ADDR})  "
          f"ports: plain {lab.PORT_PLAIN}, tls {lab.PORT_TLS}")
        w(self.style.WARNING("  TEST-ONLY. Genuinely different stack; hash-disjoint from every prior corpus."))

        the_lab = lab.IvLab(workdir=out / ".lab").setup()
        w(f"  vsftpd up + verified on {lab.SERVER_ADDR}; capturing on {lab.VETH_H} (non-loopback).")

        specs = lab.specs()
        if opts["limit"]:
            specs = specs[:opts["limit"]]
        metas, counters = [], {"benign": 0, "ftpbf": 0}
        try:
            for spec in specs:
                metas.append(self._capture(w, spec, out, counters))
        finally:
            the_lab.stop(); w("  lab stopped (servers + netns torn down).")

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
        tag = "ftps" if spec.encrypted else "plain"
        pcap = dest / f"{cap_id}_{tag}_{spec.scenario}.pcap"
        cap = lab.Capture(pcap_path=pcap); ts = datetime.now(timezone.utc).isoformat()
        cap.start()
        try:
            stats = spec.fn(None)
        except Exception as e:  # noqa: BLE001 - client-side error still leaves real captured traffic
            stats = {"attempts": 0, "client_error": str(e)}
        finally:
            cap.stop()
        chk = lab.verify_pcap(pcap)
        status = "valid" if chk.ok else "INVALID:" + ";".join(chk.errors)
        port = lab.PORT_TLS if spec.encrypted else lab.PORT_PLAIN
        meta = lab.CaptureMeta(
            capture_id=cap_id, scenario=spec.scenario, label=spec.label, client=spec.client,
            server="vsftpd" + ("(tls)" if spec.encrypted else "(plaintext)"),
            environment="netns-veth-" + ("ftps" if spec.encrypted else "plaintext"),
            interface=lab.VETH_H, mode=spec.mode, scenario_family=spec.family, encrypted=spec.encrypted,
            attempts=int(stats.get("attempts", 0)), duration_s=round(chk.duration_s, 4),
            packet_count=chk.packets, capture_timestamp=ts, source=lab.HOST_ADDR,
            destination=f"{lab.SERVER_ADDR}:{port}",
            capture_command=f"tcpdump -i {lab.VETH_H} -w <pcap> -U -n '{lab.bpf()}'",
            verification_status=status, detail={**stats, "check_errors": chk.errors})
        (out / "metadata" / f"{cap_id}.json").write_text(json.dumps({**meta.as_row(), "detail": meta.detail}, indent=2, default=str))
        flag = self.style.SUCCESS("ok") if chk.ok else self.style.ERROR("FAIL")
        w(f"  [{flag}] {cap_id:10} {spec.family:16} {spec.scenario:26} {spec.client:14} pkts={chk.packets:4} dur={chk.duration_s:5.2f}")
        return meta

    def _extract(self, w, metas, out):
        from predictor import pcap_validation as pv, ftp_behavioral as fb, live_capture
        live_capture._ensure_live_on_path()
        w(self.style.MIGRATE_HEADING("\nFeature extraction (existing pipeline, unchanged)"))
        rows, total, bad = [], 0, 0
        for m in metas:
            folder = "benign" if m.label == "Benign" else "ftp_bruteforce"
            pcap = next((out / folder).glob(f"{m.capture_id}_*.pcap"))
            behav = fb.behavioural_features_for_pcap(pcap)
            n = ok = inc = 0
            try:
                flows = pv.replay_pcap(pcap)
            except Exception as e:  # noqa: BLE001
                rows.append({"capture_id": m.capture_id, "flows": 0, "valid_flows": 0, "incomplete_flows": 0,
                             "encrypted": m.encrypted, "extraction_error": str(e)}); continue
            for fl in flows:
                n += 1
                if pv._feature_problem(fl["features"]):
                    inc += 1
                else:
                    ok += 1
            total += n; bad += inc
            rows.append({"capture_id": m.capture_id, "label": m.label, "encrypted": m.encrypted,
                         "flows": n, "valid_flows": ok, "incomplete_flows": inc,
                         "ftp_login_attempts": behav["ftp_login_attempts"],
                         "ftp_failed_logins": behav["ftp_failed_logins"],
                         "ftp_has_successful_auth": behav["ftp_has_successful_auth"], "extraction_error": ""})
        pd.DataFrame(rows).to_csv(out / "feature_extraction_report.csv", index=False)
        w(f"  total flows: {total}, incomplete (reported, not zero-filled): {bad}")
        return {"total_flows": int(total), "total_incomplete": int(bad), "n_features": len(ml.FEATURES)}

    def _leakage(self, w, root, out, metas):
        new = {_sha(p) for p in out.rglob("*.pcap")}
        others = {}
        for name, d in (("v1", "realistic_pcaps"), ("v2", "realistic_pcaps_v2"),
                        ("independent", "independent_real_pcaps"), ("targeted", "targeted_benign_pcaps"),
                        ("robustness", "robustness_pcaps"), ("robust_train", "robust_train_pcaps")):
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
        enc = [m for m in metas if m.encrypted]

        def dist(key):
            return dict(Counter(getattr(m, key) for m in metas))

        summary = {"experiment": "independent_ftp_validation_corpus", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "purpose": "Independent test of the robust behavioural candidate on a genuinely different stack.",
                   "test_only": True, "labels_from": "scenario folder (never predictions)",
                   "server": "vsFTPd (real; not used in any prior corpus)", "client_new": "lftp",
                   "network": f"veth + ip netns {lab.LAB_NET} (non-loopback, server {lab.SERVER_ADDR})",
                   "pcap_count": len(metas), "benign_count": len(benign), "ftp_bruteforce_count": len(bf),
                   "encrypted_ftps_count": len(enc),
                   "captures_passing_verification": sum(m.verification_status == "valid" for m in metas),
                   "total_flows": extraction["total_flows"], "total_incomplete_flows": extraction["total_incomplete"],
                   "diversity": {"by_scenario_family": dist("scenario_family"), "by_client": dist("client"),
                                 "by_server": dist("server"), "by_mode": dist("mode"), "by_environment": dist("environment")},
                   "leakage": leak,
                   "limitations": ["Single container: cannot run different OS or physical hosts; the netns provides a "
                                   "separate network stack + non-loopback interface, not a separate machine.",
                                   "One real server implementation (vsftpd) and cleartext + one FTPS config.",
                                   "FTPS captures: behavioural features are genuinely unavailable (encrypted control channel).",
                                   "Test-only corpus; never used for training/selection/tuning."],
                   "versions": {"python": platform.python_version(),
                                "vsftpd": subprocess.run(["/usr/sbin/vsftpd", "-v", "0"], capture_output=True, text=True).stderr.strip()
                                or subprocess.run(["dpkg-query", "-W", "-f=${Version}", "vsftpd"], capture_output=True, text=True).stdout.strip(),
                                "lftp": subprocess.run(["lftp", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]}}
        (out / "collection_report.json").write_text(json.dumps(summary, indent=2, default=str))
        lines = "\n".join(f"| {m.capture_id} | {m.scenario} | {m.scenario_family} | {m.label} | {m.client} | {m.server} | {'yes' if m.encrypted else 'no'} | {m.packet_count} | {m.verification_status} |" for m in metas)
        (out / "collection_report.md").write_text(f"""# Independent FTP validation corpus - collection report

**TEST-ONLY. Genuinely different stack: real vsFTPd server, lftp client, non-loopback
private network (veth + ip netns {lab.LAB_NET}), plus separate FTPS/TLS.** Fresh
scenario code; labels from the scenario folder only; hash-disjoint from v1/v2/
independent/targeted/robustness/robust_train (leakage all_pass={leak['all_pass']}).

## Totals
- PCAPs: **{len(metas)}** ({len(benign)} benign, {len(bf)} FTP-bruteforce; {len(enc)} FTPS/encrypted); verified {summary['captures_passing_verification']}/{len(metas)}
- Flows: {extraction['total_flows']} (incomplete: {extraction['total_incomplete']})

## What makes this independent
- **Server**: vsFTPd (real production daemon) - never used in any prior corpus (which used pyftpdlib + a bespoke raw-socket server).
- **Client**: lftp (new), plus curl / python-ftplib / raw-socket.
- **Network**: a real veth pair to an isolated `ip netns`; traffic traverses `{lab.VETH_H}` (non-loopback), addresses in {lab.LAB_NET}.
- **FTPS/TLS**: separate encrypted captures - the cleartext behavioural features are genuinely unavailable.

## Captures
| id | scenario | family | label | client | server | ftps | pkts | verify |
|---|---|---|---|---|---|---|---|---|
{lines}

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
""")
