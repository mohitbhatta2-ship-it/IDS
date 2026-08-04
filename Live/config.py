"""
Shared configuration for the live traffic module.

Everything that used to be a hardcoded Windows path lives here, resolved either
from an environment variable or by searching the repository, so the module runs
on any machine from a fresh clone.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Locating the data bundle
# ---------------------------------------------------------------------------


def locate_data_root() -> Path:
    """
    Find the folder holding Processed_Data/ and Results/Models/.

    Order of preference:
      1. $NIDS_PROJECT_PATH
      2. webapp_data/ or c_filesnew/ somewhere at or above this file
    """
    env = os.environ.get("NIDS_PROJECT_PATH")
    if env:
        return Path(env)

    here = Path(__file__).resolve()
    for parent in here.parents:
        for name in ("webapp_data", "c_filesnew"):
            candidate = parent / name
            if (candidate / "Results" / "Models").is_dir():
                return candidate

    raise FileNotFoundError(
        "Could not find the data bundle.\n"
        "Set NIDS_PROJECT_PATH to the folder containing Processed_Data/ and "
        "Results/Models/, for example:\n"
        "    export NIDS_PROJECT_PATH=~/IDS/webapp_data\n"
        "If you have just cloned the repo the models are in Git LFS -- run:\n"
        "    git lfs install && git lfs pull"
    )


DATA_ROOT = locate_data_root()

PROCESSED_DATA_PATH = DATA_ROOT / "Processed_Data"
MODELS_PATH = DATA_ROOT / "Results" / "Models"

FEATURES_FILE = PROCESSED_DATA_PATH / "selected_features.pkl"
LABEL_MAP_FILE = PROCESSED_DATA_PATH / "label_mapping.csv"

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
# The original code loaded RandomForest_Tuned.pkl, which is excluded by
# .gitignore and is not in the repository -- and is also the weakest of the four
# models (macro F1 0.8085 against HistGradientBoosting's 0.8627). Default to the
# best one that is actually committed. Override with $NIDS_MODEL.

MODELS = {
    "histgradientboosting": ("HistGradientBoosting_Tuned.pkl", None),
    "xgboost": ("XGBoost_Tuned.pkl", None),
    "mlp": ("MLP_Tuned.pkl", "MLP_StandardScaler.pkl"),
}

MODEL_NAME = os.environ.get("NIDS_MODEL", "histgradientboosting").lower()

if MODEL_NAME not in MODELS:
    raise ValueError(
        f"Unknown NIDS_MODEL={MODEL_NAME!r}. Choose one of: {', '.join(MODELS)}"
    )

_model_file, _scaler_file = MODELS[MODEL_NAME]
MODEL_FILE = MODELS_PATH / _model_file
SCALER_FILE = MODELS_PATH / _scaler_file if _scaler_file else None

# ---------------------------------------------------------------------------
# Capture behaviour
# ---------------------------------------------------------------------------

# CICFlowMeter -- which produced the training features -- uses 120 s. Matching it
# matters: the timeout decides where flows are cut, and therefore Flow Duration
# and every inter-arrival statistic derived from it.
FLOW_TIMEOUT = float(os.environ.get("NIDS_FLOW_TIMEOUT", "120"))

# How often to sweep for expired flows. The original code swept on every packet,
# which is O(flows) per packet.
CLEANUP_INTERVAL = float(os.environ.get("NIDS_CLEANUP_INTERVAL", "5"))

# Hard cap on tracked flows so a long capture cannot exhaust memory.
MAX_FLOWS = int(os.environ.get("NIDS_MAX_FLOWS", "20000"))

# Where predicted flows are appended.
CSV_FILE = Path(os.environ.get("NIDS_OUTPUT_CSV", "live_flows.csv"))


def default_interface() -> str | None:
    """
    Capture interface. $NIDS_INTERFACE wins; otherwise let scapy decide, which
    is right far more often than the hardcoded Windows name "Wi-Fi" was.
    """
    iface = os.environ.get("NIDS_INTERFACE")
    if iface:
        return iface
    return None


def describe() -> str:
    return (
        f"data root   : {DATA_ROOT}\n"
        f"model       : {MODEL_NAME} ({MODEL_FILE.name})\n"
        f"flow timeout: {FLOW_TIMEOUT:.0f}s\n"
        f"output csv  : {CSV_FILE}"
    )
