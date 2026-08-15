# Real-PCAP validation workflow

Measure, honestly, how the **existing saved model** generalises to realistically
captured traffic — without retraining, tuning, thresholds, or forced
predictions. Captured packets flow through the *exact* Live Capture flow engine
and the *exact* Dataset Testing model path; the only new thing is that ground
truth comes from the capture you labelled.

```
PCAP → flow construction (Live Capture) → 30 features → preprocessing
     → existing model → prediction → compare to known label → metrics
```

## Safety first

Only capture traffic and run attack tools against **systems and networks you own
or are explicitly authorised to test** (localhost, a VM, or a lab host on a
private network). Do not point any of this at third-party or shared systems.

## The pipeline (what actually runs)

| Stage | Reused implementation |
|---|---|
| Packets → flows | `predictor.live_capture.CaptureSession` (bidirectional 5-tuple, TCP FIN/RST finalisation, 120 s idle timeout, end-of-capture flush) |
| Flows → 30 features | `Live/feature_calculator.calculate_features` (the single implementation) |
| Preprocessing + model + evaluation | `predictor.ml.predict_batch` — i.e. **Dataset Testing itself** |

The pcap path converges into `predict_batch`, so preprocessing parity is
structural, not asserted. Features are validated to be exactly the 30
`ml.FEATURES`, all finite; incomplete flows are reported, never zero-filled.

## Step by step

### 1. Start a controlled target you own
```bash
python3 -m http.server 8080        # web target for HTTP DoS
# or your own sshd (:22) / vsftpd (:21) on a VM/localhost
```

### 2. Start packet capture (one class per file, use a filter to stay on-target)
```bash
# capture only the attack's traffic so the pcap is cleanly labelled
sudo tcpdump -i lo -w validation/pcaps/ftp_bruteforce/session1.pcap 'tcp port 21'
```
On WSL/Windows, capture the interface the traffic actually crosses (e.g. the
WSL `vEthernet` bridge for Windows↔WSL traffic, or `lo` for localhost).

### 3. Generate ONE controlled attack against your own service
| Class (folder) | Example against your own host |
|---|---|
| `benign` | `curl`, a browser session, a file download (baseline) |
| `ftp_bruteforce` | `hydra -l test -P words.txt ftp://127.0.0.1` |
| `ssh_bruteforce` | `hydra -l test -P words.txt ssh://127.0.0.1` |
| `dos_hulk` | `python3 hulk.py http://127.0.0.1:8080` |
| `dos_slowloris` | `slowloris 127.0.0.1 -p 8080` |

Always capture a **Benign baseline** too — a validation that only reacts to
attacks but flags normal traffic as attack is not a pass.

### 4. Stop capture
`Ctrl-C` the `tcpdump`. One capture = one attack session = one label.

### 5. Label the PCAP
Labelling is just the folder: put the file under the matching `validation/pcaps/<class>/`.

### 6. Run the validation workflow (from `webapp_django/`)
```bash
# whole tree, with the CIC-vs-real comparison, writing CSV + JSON
python manage.py validate_pcaps --input ../validation/pcaps \
    --output ../validation/results --compare-cic

# or a single file
python manage.py validate_pcaps \
    --pcap ../validation/pcaps/ftp_bruteforce/session1.pcap --label FTP-BruteForce
```

### 7. Inspect the metrics and confusion matrix
The command prints the per-class table (Class | Ground Truth | Correct |
Incorrect | Precision | Recall | F1), the confusion matrix, the
CIC-vs-real comparison, and real-vs-CIC feature medians. With `--output` it also
writes `validation/results/`:

- `flows.csv` — every flow: 5-tuple, 30 features, ground-truth Label, Predicted Class, Confidence
- `per_class.csv` — the per-class table
- `confusion_matrix.csv`
- `feature_distribution.csv` — real vs CIC medians per class/feature
- `summary.json` — accuracy, macro/weighted-F1, per-class, confusion, per-pcap counts, CIC baseline

## Reproduce the committed real-capture baseline

Real captures from our controlled WSL environment (Windows `172.24.48.1` ↔ WSL
vsftpd `172.24.60.225:21`) live in `sample_data/real_pcap/`:

```
sample_data/real_pcap/benign/{benign_01,benign_03}.pcap          -> Benign  (230 Login successful)
sample_data/real_pcap/ftp_bruteforce/{ftp_01,ftp_02,ftp_03}.pcap -> FTP-BruteForce (repeated USER/PASS -> 530)
```

Run the whole baseline (from `webapp_django/`):

```bash
python manage.py validate_pcaps \
    --input ../sample_data/real_pcap \
    --output ../validation/results \
    --compare-cic
```

Committed result (`validation/results/`), existing model, no retraining:

| | Flows | Benign correct | FTP-BruteForce correct | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| **Real capture** | 42 | 8 / 8 | **0 / 34** | 0.190 | 0.160 |
| **CIC held-out (Dataset Testing)** | 40,000 | — | — | 0.980 | 0.883 |

Confusion matrix: every FTP-BruteForce flow → Benign. The model does **not**
generalise to real FTP brute force (0 % recall), while remaining healthy on CIC.

**Flow counts.** The pipeline (identical to Live Capture) produces ~2 flows per
FTP connection: one substantive login flow (6–21 packets, median duration
~3.1 s) and one short post-FIN residual flow (1–2 packets). Both are classified
Benign. This residual-flow behaviour mirrors Live Capture and is left unchanged.

**Why Benign.** The real substantive FTP login flow is a multi-second, low-rate,
small-payload connection (median Flow Duration ~3.1 s, Flow Pkts/s ~3.8,
TotLen Fwd ~27 B). The CIC FTP-BruteForce class is the opposite — a ~4 µs,
zero-payload, ~225 k pkts/s artifact. The two occupy completely different
feature regions, so a correctly-extracted real login lands in Benign.

## Ground truth & leakage

- Ground truth is the capture's label, never the model's prediction.
- Each session stays in its own file; flows from one session are never split
  across an evaluation boundary. We are **not** training, so there is no
  train/test split to leak across — the whole capture is scored as-is.

## What this can and cannot show

Some CIC-IDS2018 attack classes are defined by capture artifacts real traffic
cannot reproduce (e.g. FTP-BruteForce training flows are ~4 µs, zero-payload,
~225k pkts/s). A correctly-extracted real FTP login therefore looks Benign — a
generalisation limit of the trained model/data, not a bug in capture. This
workflow quantifies exactly that gap so a later decision about realistic-traffic
retraining can be made on evidence.
