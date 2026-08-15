# Real-PCAP validation data

Drop **real, self-captured** PCAPs here to measure how the existing model
generalises to realistic traffic. Nothing here is used to train — this is an
honest, read-only baseline.

```
validation/
  pcaps/
    benign/            <- normal traffic (baseline)
    ftp_bruteforce/    <- FTP brute force against YOUR OWN server
    ssh_bruteforce/    <- SSH brute force against YOUR OWN server
    dos_hulk/          <- HTTP flood (Hulk) against YOUR OWN server
    dos_slowloris/     <- slow-connection DoS against YOUR OWN server
  features/            <- optional intermediate feature dumps
  results/             <- metrics written here (CSV + JSON)
```

**Ground truth = the folder name.** Every flow in `ftp_bruteforce/*.pcap` is
labelled `FTP-BruteForce`, and so on. The model's prediction is never used as
truth. Keep each attack session in its own file; sessions are never mixed.

Folder → class mapping: `benign → Benign`, `ftp_bruteforce → FTP-BruteForce`,
`ssh_bruteforce → SSH-Bruteforce`, `dos_hulk → DoS attacks-Hulk`,
`dos_slowloris → DoS attacks-Slowloris`. A bare `dos/` is intentionally rejected
— name the specific DoS class.

Run it (from `webapp_django/`):

```bash
# committed real captures live in sample_data/real_pcap/
python manage.py validate_pcaps --input ../sample_data/real_pcap --output ../validation/results --compare-cic

# your own drop-in captures under validation/pcaps/
python manage.py validate_pcaps --input ../validation/pcaps --output ../validation/results --compare-cic

# or a single capture:
python manage.py validate_pcaps --pcap ../sample_data/real_pcap/ftp_bruteforce/ftp_01.pcap --label FTP-BruteForce
```

**Currently validated classes:** Benign, FTP-BruteForce (real captures present).
**Not yet validated:** SSH-Bruteforce, DoS/HTTP — no real captures available yet
(drop them into the matching folder to validate; no fake PCAPs are created).

Full, safety-first instructions: [`docs/real-pcap-validation.md`](../docs/real-pcap-validation.md).

> Only capture traffic and generate attacks on systems and networks you own or
> are explicitly authorised to test. The `.gitkeep` files just keep the empty
> folders in git; no capture data is committed.
