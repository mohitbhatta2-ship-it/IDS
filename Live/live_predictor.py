"""
Classifies a preprocessed flow.

The model, the optional scaler and the label mapping are all resolved through
config.py rather than hardcoded paths, and the model is loaded lazily so simply
importing this module does not cost several seconds or fail on a machine where
the .pkl files have not been fetched from Git LFS.
"""

import joblib
import pandas as pd

import config

_model = None
_scaler = None
_label_map = None


def _require(path, what):
    if not path.is_file():
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            "The model files are stored in Git LFS. If this is a fresh clone, run:\n"
            "    git lfs install && git lfs pull"
        )
    return path


def load():
    """Load the model, scaler and label map once, then reuse them."""
    global _model, _scaler, _label_map

    if _model is not None:
        return _model, _scaler, _label_map

    _model = joblib.load(_require(config.MODEL_FILE, "Model"))

    if config.SCALER_FILE is not None:
        _scaler = joblib.load(_require(config.SCALER_FILE, "Scaler"))

    mapping = pd.read_csv(_require(config.LABEL_MAP_FILE, "Label mapping"))
    _label_map = dict(zip(mapping["Encoded"], mapping["Class"]))

    return _model, _scaler, _label_map


def predict(df, top_n=3):
    """
    Classify one preprocessed flow.

    Returns a list of (label, probability) ordered most likely first. Callers
    that want a single verdict should take element 0 -- note this is a LIST, not
    a 2-tuple, which is what flow_manager used to get wrong.
    """
    model, scaler, label_map = load()

    if scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    probabilities = model.predict_proba(df)[0]

    ranked = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return [
        (label_map.get(int(cls), f"Class {cls}"), float(prob))
        for cls, prob in ranked
    ]


def predict_one(df):
    """Convenience wrapper returning just the single best (label, probability)."""
    return predict(df, top_n=1)[0]


if __name__ == "__main__":
    load()
    print(config.describe())
    print(f"classes     : {len(_label_map)}")
