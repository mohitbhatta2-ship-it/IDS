"""
CSE-CIC-IDS2018 NIDS Project — Results Dashboard
--------------------------------------------------
Run locally with:
    streamlit run app.py

Update PROJECT_PATH below to match your machine if it differs.
"""

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG — update this path if your folder structure differs
# ---------------------------------------------------------------------------
import os

# Locally on Windows: set NIDS_PROJECT_PATH once (e.g. in your terminal before running),
# or just leave it unset -- it falls back to your usual local path below.
# On Hugging Face Spaces: leave the env var unset entirely -- it automatically falls back
# to a "data" folder placed right next to this script in the repo.
_DEFAULT_LOCAL_PATH = r"M:\IDS\c_filesnew"
_SCRIPT_RELATIVE_DATA = Path(__file__).parent / "data"

if os.environ.get("NIDS_PROJECT_PATH"):
    PROJECT_PATH = Path(os.environ["NIDS_PROJECT_PATH"])
elif Path(_DEFAULT_LOCAL_PATH).exists():
    PROJECT_PATH = Path(_DEFAULT_LOCAL_PATH)
else:
    PROJECT_PATH = _SCRIPT_RELATIVE_DATA

PROCESSED_DATA_PATH = PROJECT_PATH / "Processed_Data"
RESULTS_PATH = PROJECT_PATH / "Results"
MODELS_PATH = RESULTS_PATH / "Models"
FIGURES_PATH = RESULTS_PATH / "Figures"
TABLES_PATH = RESULTS_PATH / "Tables"

st.set_page_config(page_title="NIDS Project Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def show_image_if_exists(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption, width='stretch')
    else:
        st.info(f"Figure not found: `{path.name}` — run the corresponding notebook cell to generate it.")


def show_csv_if_exists(path: Path, **kwargs):
    if path.exists():
        df = pd.read_csv(path)
        st.dataframe(df, width='stretch', **kwargs)
        return df
    st.info(f"Table not found: `{path.name}` — run the corresponding notebook cell to generate it.")
    return None


@st.cache_data
def load_label_mapping():
    path = PROCESSED_DATA_PATH / "label_mapping.csv"
    if not path.exists():
        return None, None, None
    mapping = pd.read_csv(path)
    label_to_id = dict(zip(mapping["Class"], mapping["Encoded"]))
    id_to_label = dict(zip(mapping["Encoded"], mapping["Class"]))
    return mapping, label_to_id, id_to_label


@st.cache_data
def load_selected_features():
    path = PROCESSED_DATA_PATH / "selected_features.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_feature_reference_stats():
    """Min/max/percentiles per feature from the real (balanced) training data —
    used to (a) generate random values in a realistic range and (b) flag how
    out-of-distribution a generated sample is."""
    path = PROCESSED_DATA_PATH / "balanced_train_selected.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    features = load_selected_features() or [c for c in df.columns if c != "Label"]
    stats = {}
    for col in features:
        if col not in df.columns:
            continue
        series = df[col]
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "p01": float(series.quantile(0.01)),
            "p99": float(series.quantile(0.99)),
            "p10": float(series.quantile(0.10)),
            "p90": float(series.quantile(0.90)),
            "is_categorical": col in ("Dst Port", "Protocol"),
            "categories": sorted(series.unique().tolist()) if col in ("Dst Port", "Protocol") else None,
        }
    return stats


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


def discover_tuned_models():
    """Finds every *_Tuned.pkl in Results/Models/."""
    if not MODELS_PATH.exists():
        return []
    return sorted(MODELS_PATH.glob("*_Tuned.pkl"))


@st.cache_data
def load_test_data():
    """The real, held-out test set -- never used in training or CTGAN augmentation."""
    path = PROCESSED_DATA_PATH / "test_selected.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def run_all_models(X_row_base: pd.DataFrame, tuned_model_paths, id_to_label: dict):
    """Runs a single-row DataFrame through every tuned model, handling MLP scaling and
    per-model feature reindexing. Shared by both the random-input and real-test-sample modes."""
    predictions = {}
    skipped_models = []

    for model_path in tuned_model_paths:
        model_name = model_path.stem
        model = load_model(model_path)

        is_mlp = model_name.upper().startswith("MLP")
        scaler = None
        if is_mlp:
            scaler_path = MODELS_PATH / "MLP_StandardScaler.pkl"
            if scaler_path.exists():
                scaler = load_model(scaler_path)

        reference_obj = scaler if scaler is not None else model
        expected_cols = getattr(reference_obj, "feature_names_in_", None)

        if expected_cols is not None:
            missing = [c for c in expected_cols if c not in X_row_base.columns]
            if missing:
                skipped_models.append((model_name, missing))
                continue
            X_row = X_row_base.reindex(columns=expected_cols)
        else:
            X_row = X_row_base

        if scaler is not None:
            X_row = pd.DataFrame(scaler.transform(X_row), columns=X_row.columns)

        pred_encoded = model.predict(X_row)[0]
        pred_label = id_to_label.get(pred_encoded, str(pred_encoded))
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(X_row)[0].max())

        predictions[model_name] = {"prediction": pred_label, "confidence": confidence}

    return predictions, skipped_models


def compute_mlp_activations(model, X_scaled: np.ndarray):
    """Manual forward pass through an MLPClassifier's real trained weights.
    Verified to match model.predict_proba() exactly (output layer uses softmax
    for multiclass, matching sklearn's actual internal behavior).
    Returns a list of activation arrays: [input, hidden_1, ..., hidden_k, output_probs]."""
    activations = [X_scaled]
    n_layers = len(model.coefs_)
    for i in range(n_layers):
        Z = activations[-1] @ model.coefs_[i] + model.intercepts_[i]
        if i == n_layers - 1:
            expZ = np.exp(Z - Z.max(axis=1, keepdims=True))
            A = expZ / expZ.sum(axis=1, keepdims=True)
        elif model.activation == "relu":
            A = np.maximum(Z, 0)
        elif model.activation == "tanh":
            A = np.tanh(Z)
        elif model.activation == "logistic":
            A = 1 / (1 + np.exp(-Z))
        else:
            A = Z
        activations.append(A)
    return activations


@st.cache_resource
def get_mlp_and_scaler():
    mlp_path = MODELS_PATH / "MLP_Tuned.pkl"
    scaler_path = MODELS_PATH / "MLP_StandardScaler.pkl"
    if not mlp_path.exists() or not scaler_path.exists():
        return None, None
    return load_model(mlp_path), load_model(scaler_path)


@st.cache_data
def compute_neuron_class_activations(_model, _scaler, sample_size=3000):
    """Runs a sample of the real test set through the MLP and returns, for every hidden
    neuron, its mean activation per true class -- this is what the Neuron Inspector shows."""
    test_data = load_test_data()
    selected_features = load_selected_features()
    if test_data is None or selected_features is None:
        return None

    sample = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)
    X = sample[selected_features]
    y_true = sample["Label"].values

    X_scaled = _scaler.transform(X)
    activations = compute_mlp_activations(_model, X_scaled)

    _, _, id_to_label = load_label_mapping()
    classes_present = sorted(set(y_true))
    class_names = [id_to_label.get(c, str(c)) for c in classes_present]

    # for each hidden layer, mean activation per neuron per class
    hidden_layer_class_means = []
    for layer_activations in activations[1:-1]:  # skip input and output layers
        layer_means = np.array([
            layer_activations[y_true == c].mean(axis=0) for c in classes_present
        ])  # shape: (n_classes, n_neurons)
        hidden_layer_class_means.append(layer_means)

    overall_stats = [
        (layer_activations.mean(axis=0), layer_activations.std(axis=0))
        for layer_activations in activations[1:-1]
    ]

    return hidden_layer_class_means, overall_stats, class_names


def safe_name_to_class_map(label_to_id: dict):
    return {str(cls).replace("/", "_").replace(" ", "_"): cls for cls in label_to_id.keys()}


def discover_ctgan_class_generators():
    """Finds every saved per-class CTGAN generator model."""
    ctgan_models_path = PROJECT_PATH / "CTGAN" / "Models"
    if not ctgan_models_path.exists():
        return {}
    return {f.stem.replace("_ctgan", ""): f for f in ctgan_models_path.glob("*_ctgan.pkl")}


def generate_synthetic_class_sample(safe_name: str, count: int):
    """Samples `count` synthetic rows from a saved per-class CTGAN generator, applying the
    same log1p inversion and non-negative clipping used during training (Section 5 of
    03_Data_Preparation_for_CTGAN.ipynb), so generated samples are consistent with what
    actually went into training."""
    ctgan_models_path = PROJECT_PATH / "CTGAN" / "Models"
    generator = load_model(ctgan_models_path / f"{safe_name}_ctgan.pkl")

    log_columns_path = ctgan_models_path / f"{safe_name}_log_columns.json"
    log_columns = []
    if log_columns_path.exists():
        with open(log_columns_path) as f:
            log_columns = json.load(f)

    synthetic = generator.sample(count)
    for col in log_columns:
        if col in synthetic.columns:
            synthetic[col] = np.expm1(synthetic[col])

    numeric_cols = synthetic.select_dtypes(include=np.number).columns
    non_negative_cols = [c for c in numeric_cols if c not in ("Dst Port", "Protocol")]
    synthetic[non_negative_cols] = synthetic[non_negative_cols].clip(lower=0)

    return synthetic


def generate_random_flow(stats: dict):
    """Uniform-random value per feature, within that feature's real observed range.
    Categorical columns (Dst Port, Protocol) sample from actual observed categories
    instead of a meaningless continuous range."""
    row = {}
    for col, s in stats.items():
        if s["is_categorical"]:
            row[col] = np.random.choice(s["categories"])
        else:
            row[col] = np.random.uniform(s["p01"], s["p99"])
    return row


def ood_flags(row: dict, stats: dict):
    """Which features in this generated row fall outside the 10th-90th percentile
    of real observed values — i.e. values that, while within the (wider) generation
    range, are still atypical compared to where most real traffic actually sits."""
    flags = []
    for col, value in row.items():
        s = stats.get(col)
        if not s or s["is_categorical"]:
            continue
        if value < s["p10"] or value > s["p90"]:
            flags.append(col)
    return flags


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("CSE-CIC-IDS2018 — Network Intrusion Detection Pipeline")
st.caption(
    "Data cleaning → CTGAN-based minority-class augmentation → Optuna-driven model "
    "selection (5 algorithms) → independently-reported MLP baseline."
)

tabs = st.tabs([
    "Overview",
    "Model Comparison",
    "Rare-Class Performance",
    "CTGAN Diagnostics",
    "Optuna Search",
    "Try It Yourself",
    "🧠 NN Playground",
])

# ---------------------------------------------------------------------------
# TAB 1 — Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Pipeline Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pipeline Stages")
        st.markdown("""
1. **Data Cleaning** — chunked processing of raw CICIDS2018 CSVs, schema verification,
   missing/infinite-value handling, DuckDB merge & deduplication
2. **Stratified Sampling** — built a manageable, class-aware dataset, keeping every rare-class row
3. **CTGAN Augmentation** — dual-path strategy: bootstrap + jitter for ultra-rare classes
   (< 1,000 real samples), ratio-capped CTGAN for the rest — avoids fabricating classes
   almost entirely from synthetic data
4. **Majority-Class Downsampling** — Benign capped at 100,000 in training data only
   (test set stays representative of real-world proportions)
5. **Optuna Model Selection** — single search space across DecisionTree, RandomForest,
   HistGradientBoosting, XGBoost, AdaBoost; top 3 retrained on full data
6. **MLP Baseline** — separately tuned, reported independently as a deep-learning reference point
        """)

        with st.expander("⚠️ Known issue found & fixed: data leakage via `rn` column"):
            st.markdown("""
A DuckDB stratified-sampling query (Section 7 of the data prep notebook) used
`ROW_NUMBER() OVER (PARTITION BY Label ORDER BY random()) AS rn` as a temporary sampling helper,
but `SELECT d.*` accidentally carried that column into the final dataset — where it was then
selected as one of the 30 model input features by mutual information, purely because it
correlated with class labels as a side effect of per-class sampling quotas, not real network
behavior.

**Impact once removed:** Macro F1 dropped across every model (most notably ~10 points for MLP),
confirming the leak had been inflating rare-class performance specifically. Fixed via
`SELECT d.* EXCLUDE (rn)` and a full re-run of the downstream pipeline. All results shown in this
dashboard are from the corrected pipeline.
            """)

    with col2:
        st.subheader("Headline Results")
        metrics_files = sorted(MODELS_PATH.glob("*_metrics.csv")) if MODELS_PATH.exists() else []
        if metrics_files:
            all_metrics = pd.concat([pd.read_csv(f) for f in metrics_files], ignore_index=True)
            all_metrics = all_metrics.sort_values("Macro F1", ascending=False).reset_index(drop=True)

            best = all_metrics.iloc[0]
            st.metric("Best Model", best["Model"])
            m1, m2 = st.columns(2)
            m1.metric("Test Accuracy", f"{best['Accuracy']:.4f}")
            m2.metric("Test Macro F1", f"{best['Macro F1']:.4f}")

            st.caption(f"{len(all_metrics)} models trained and compared:")
            st.dataframe(
                all_metrics[["Model", "Accuracy", "Macro F1", "Weighted F1"]],
                width='stretch', hide_index=True,
            )
        else:
            st.info("No trained models found yet — run notebooks 04 and 06 first.")

# ---------------------------------------------------------------------------
# TAB 2 — Model Comparison
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Model Comparison")
    st.markdown(
        "All trained models, discovered automatically from `Results/Models/*_metrics.csv` — "
        "including the Optuna-selected top 3 and the independently-tuned MLP baseline."
    )

    metrics_files = sorted(MODELS_PATH.glob("*_metrics.csv")) if MODELS_PATH.exists() else []
    if metrics_files:
        all_metrics = pd.concat([pd.read_csv(f) for f in metrics_files], ignore_index=True)
        all_metrics = all_metrics.sort_values("Macro F1", ascending=False).reset_index(drop=True)
        st.dataframe(all_metrics, width='stretch', hide_index=True)
    else:
        st.info("No model metrics found yet.")

    col1, col2 = st.columns(2)
    with col1:
        show_image_if_exists(FIGURES_PATH / "accuracy_comparison.png", "Accuracy Comparison")
    with col2:
        show_image_if_exists(FIGURES_PATH / "macro_f1_comparison.png", "Macro F1 Comparison")

    show_image_if_exists(FIGURES_PATH / "precision_recall_f1_comparison.png", "Precision / Recall / F1")
    show_image_if_exists(FIGURES_PATH / "performance_heatmap.png", "Performance Heatmap")

    st.subheader("Train vs. Test Gap (Overfitting Check)")
    show_csv_if_exists(TABLES_PATH / "train_vs_test_gap.csv", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3 — Rare-Class Performance
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Rare-Class Performance")
    st.markdown(
        "Overall accuracy is dominated by common classes like Benign. This view isolates "
        "performance on the classes that matter most for a security use case but are hardest "
        "to learn: the ones with very few real samples."
    )
    show_image_if_exists(FIGURES_PATH / "rare_class_f1_comparison.png", "F1-Score on Rare Classes, by Model")
    show_csv_if_exists(TABLES_PATH / "rare_class_f1_comparison.csv")

# ---------------------------------------------------------------------------
# TAB 4 — CTGAN Diagnostics
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("CTGAN Augmentation Diagnostics")

    st.subheader("Class Distribution: Before vs. After Augmentation")
    show_image_if_exists(
        FIGURES_PATH / "class_distribution_before_after_CTGAN.png",
        "Class Distribution Before/After CTGAN + Bootstrap",
    )
    class_dist_path = TABLES_PATH / "class_distribution_before_after_CTGAN.csv"
    if class_dist_path.exists():
        class_dist = pd.read_csv(class_dist_path)
        # notebook 05 now saves this with the attack name already as the index/first column
        st.dataframe(class_dist, width='stretch', hide_index=True)
    else:
        st.info(
            f"Table not found: `{class_dist_path.name}` — run the corresponding notebook cell "
            f"to generate it."
        )

    st.subheader("A Worked Example: Why Not Just Force Every Class to 10,000?")
    st.markdown(
        "`Brute Force -Web` has only 489 real samples. Forcing CTGAN to generate 9,511 synthetic "
        "rows to hit a flat 10,000-row target produced synthetic data that barely resembled the "
        "real class — confirmed empirically, not assumed:"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before fix** (CTGAN forced to fill the full 10,000 target)")
        st.metric("Avg Mean Diff %", "220.74%", delta="Unreliable", delta_color="inverse")
        st.metric("Avg Std Diff %", "276.03%", delta="Unreliable", delta_color="inverse")
    with c2:
        st.markdown("**After fix** (bootstrap + jitter, since 489 < 1,000 threshold)")
        st.metric("Avg Mean Diff %", "1.46%", delta="Reliable", delta_color="normal")
        st.metric("Avg Std Diff %", "0.98%", delta="Reliable", delta_color="normal")

    st.caption(
        "Classes with ≥ 1,000 real samples use CTGAN, capped at 3× the real count "
        "(ratio-capped); classes below that use bootstrap + jitter instead."
    )

    st.subheader("Top Mutual Information Features")
    mi_path = RESULTS_PATH / "mutual_information_scores.csv"
    show_image_if_exists(FIGURES_PATH / "top20_mutual_information.png", "Top 20 Features by Mutual Information")
    show_csv_if_exists(mi_path)

# ---------------------------------------------------------------------------
# TAB 5 — Optuna Search
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Optuna Algorithm & Hyperparameter Search")
    st.markdown(
        "A single search space covering 5 algorithms (DecisionTree, RandomForest, "
        "HistGradientBoosting, XGBoost, AdaBoost) — Optuna picks both the algorithm and its "
        "hyperparameters per trial, scored on cross-validated macro F1."
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image_if_exists(FIGURES_PATH / "optuna_algorithm_comparison.png", "Best Score per Algorithm")
    with col2:
        show_image_if_exists(FIGURES_PATH / "optuna_optimization_history.png", "Optimization History")

    st.subheader("Top Trials")
    trials_df = show_csv_if_exists(TABLES_PATH / "optuna_trials.csv")
    if trials_df is not None:
        completed = trials_df[trials_df["state"] == "COMPLETE"].sort_values("value", ascending=False)
        st.dataframe(completed.head(15), width='stretch', hide_index=True)

    st.subheader("MLP Baseline Search")
    show_image_if_exists(FIGURES_PATH / "mlp_optuna_optimization_history.png", "MLP Optimization History")

# ---------------------------------------------------------------------------
# TAB 6 — Try It Yourself (simulator)
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Try It Yourself")

    mode = st.radio(
        "Input source",
        ["🎲 Random input (out-of-distribution check)", "✅ Real test-set flow (ground truth known)"],
        horizontal=True,
    )

    _, label_to_id, id_to_label = load_label_mapping()
    tuned_model_paths = discover_tuned_models()

    if not tuned_model_paths or id_to_label is None:
        st.info(
            "Missing required files — need `label_mapping.csv` and at least one `*_Tuned.pkl` model."
        )

    # -----------------------------------------------------------------------
    # MODE A — Random, out-of-distribution input
    # -----------------------------------------------------------------------
    elif mode.startswith("🎲"):
        st.warning(
            "**What this is (and isn't):** feature values below are drawn independently and "
            "uniformly at random from each feature's real observed range — they do **not** "
            "represent realistic network traffic (real flows have correlated features). This tests "
            "model behavior on out-of-distribution input, not detection accuracy on real attacks. "
            "For real accuracy figures, see the **Model Comparison** tab or switch to the "
            "**Real test-set flow** mode above."
        )

        feature_stats = load_feature_reference_stats()
        if feature_stats is None:
            st.info("Missing `balanced_train_selected.parquet` / `selected_features.pkl`.")
        else:
            if "agree_count" not in st.session_state:
                st.session_state.agree_count = 0
                st.session_state.total_count = 0

            if st.button("🎲 Generate Random Flow", type="primary"):
                row = generate_random_flow(feature_stats)
                flags = ood_flags(row, feature_stats)

                X_row_base = pd.DataFrame([row])
                predictions, skipped_models = run_all_models(X_row_base, tuned_model_paths, id_to_label)

                if skipped_models:
                    for name, missing in skipped_models:
                        st.error(
                            f"Skipped **{name}**: expects {len(missing)} feature(s) not present "
                            f"(e.g. {missing[:3]}...). Likely trained on a different feature set — "
                            f"re-run notebook 03 → 04/06 to regenerate consistently."
                        )

                if predictions:
                    unique_preds = set(p["prediction"] for p in predictions.values())
                    all_agree = len(unique_preds) == 1

                    st.session_state.total_count += 1
                    if all_agree:
                        st.session_state.agree_count += 1

                    if all_agree:
                        st.success(f"✅ All {len(predictions)} models agree: **{next(iter(unique_preds))}**")
                    else:
                        st.warning(f"⚠️ Models disagree: {', '.join(sorted(unique_preds))}")

                    if flags:
                        st.caption(
                            f"⚠️ {len(flags)} of {len(row)} features fall outside the typical "
                            f"10th-90th percentile range of real observed values — some "
                            f"disagreement or low confidence here is expected, not a flaw."
                        )

                    with st.expander("Show per-model details"):
                        detail_rows = [
                            {"Model": name, "Prediction": p["prediction"],
                             "Confidence": f"{p['confidence']:.3f}" if p["confidence"] is not None else "n/a"}
                            for name, p in predictions.items()
                        ]
                        st.dataframe(pd.DataFrame(detail_rows), width='stretch', hide_index=True)

                    with st.expander("Show generated feature values"):
                        st.dataframe(
                            pd.DataFrame([row]).T.rename(columns={0: "Value"}),
                            width='stretch',
                        )

            if st.session_state.total_count > 0:
                rate = st.session_state.agree_count / st.session_state.total_count * 100
                st.caption(
                    f"Session agreement rate: {st.session_state.agree_count}/"
                    f"{st.session_state.total_count} ({rate:.0f}%) — expected to be imperfect on "
                    f"random noise; this reflects model calibration, not a flaw."
                )

    # -----------------------------------------------------------------------
    # MODE B — Real test-set flow, ground truth known
    # -----------------------------------------------------------------------
    else:
        st.info(
            "Samples a real flow from the **held-out test set** — never seen during training or "
            "CTGAN augmentation. This is a genuine accuracy check, not a robustness/OOD check."
        )

        test_data = load_test_data()
        if test_data is None:
            st.info("Missing `test_selected.parquet`.")
        else:
            if "real_correct_count" not in st.session_state:
                st.session_state.real_correct_count = {}
                st.session_state.real_total_count = 0

            if st.button("✅ Sample a Real Test Flow", type="primary"):
                sample_row = test_data.sample(n=1)
                true_label_encoded = int(sample_row["Label"].iloc[0])
                true_label = id_to_label.get(true_label_encoded, str(true_label_encoded))

                X_row_base = sample_row.drop(columns=["Label"])
                predictions, skipped_models = run_all_models(X_row_base, tuned_model_paths, id_to_label)

                if skipped_models:
                    for name, missing in skipped_models:
                        st.error(f"Skipped **{name}**: expects {len(missing)} feature(s) not present.")

                if predictions:
                    st.session_state.real_total_count += 1

                    st.markdown(f"### True label: **{true_label}**")

                    result_rows = []
                    for name, p in predictions.items():
                        is_correct = p["prediction"] == true_label
                        st.session_state.real_correct_count.setdefault(name, 0)
                        if is_correct:
                            st.session_state.real_correct_count[name] += 1

                        result_rows.append({
                            "Model": name,
                            "Prediction": p["prediction"],
                            "Correct?": "✅" if is_correct else "❌",
                            "Confidence": f"{p['confidence']:.3f}" if p["confidence"] is not None else "n/a",
                        })

                    st.dataframe(pd.DataFrame(result_rows), width='stretch', hide_index=True)

                    with st.expander("Show this flow's feature values"):
                        st.dataframe(
                            X_row_base.T.rename(columns={X_row_base.index[0]: "Value"}),
                            width='stretch',
                        )

            if st.session_state.real_total_count > 0:
                st.subheader("Session accuracy per model (on sampled real test flows)")
                acc_rows = [
                    {
                        "Model": name,
                        "Correct": count,
                        "Total": st.session_state.real_total_count,
                        "Accuracy": f"{count / st.session_state.real_total_count * 100:.0f}%",
                    }
                    for name, count in st.session_state.real_correct_count.items()
                ]
                st.dataframe(pd.DataFrame(acc_rows), width='stretch', hide_index=True)
                st.caption(
                    "Small-sample accuracy here will be noisier than the full test-set metrics in "
                    "the **Model Comparison** tab — this is for intuition/demo purposes, not a "
                    "replacement for the real evaluation."
                )

# ---------------------------------------------------------------------------
# TAB 7 — Neural Network Playground
# ---------------------------------------------------------------------------
with tabs[6]:
    st.header("🧠 Neural Network Playground")
    st.caption(
        "Visualizes the **actual trained MLP's real weights and activations** — nothing here "
        "is a mockup. Node colors and the Neuron Inspector are computed from a real forward "
        "pass through `MLP_Tuned.pkl` over sampled real test-set flows."
    )

    mlp_model, mlp_scaler = get_mlp_and_scaler()

    if mlp_model is None:
        st.info("`MLP_Tuned.pkl` / `MLP_StandardScaler.pkl` not found — run notebook 06 first.")
    else:
        layer_sizes = [mlp_model.coefs_[0].shape[0]] + [c.shape[1] for c in mlp_model.coefs_]
        n_hidden_layers = len(layer_sizes) - 2
        _, _, id_to_label_nn = load_label_mapping()
        output_classes = [id_to_label_nn.get(c, str(c)) for c in mlp_model.classes_]

        result = compute_neuron_class_activations(mlp_model, mlp_scaler)
        if result is None:
            st.info("Missing `test_selected.parquet` / `selected_features.pkl`.")
        else:
            hidden_layer_class_means, overall_stats, class_names = result

            col_topo, col_inspector = st.columns([2, 1])

            with col_topo:
                st.subheader("Network Topology")
                st.caption(
                    f"Real trained architecture: {layer_sizes[0]} inputs → "
                    + " → ".join(str(s) for s in layer_sizes[1:-1])
                    + f" → {layer_sizes[-1]} outputs "
                    f"({n_hidden_layers} hidden layer{'s' if n_hidden_layers != 1 else ''})"
                )

                ctrl1, ctrl2 = st.columns(2)
                with ctrl1:
                    max_neurons_per_layer = st.slider(
                        "Max neurons shown per hidden layer", 10, 80, 30, step=5,
                        help="Dense layers (128+) are capped for legibility. The most "
                             "differentiated neurons (highest activation variance) are kept.",
                    )
                with ctrl2:
                    max_connections_per_neuron = st.slider(
                        "Max connections per neuron", 1, 5, 2,
                        help="Only the strongest real weights are drawn, per neuron.",
                    )

                # For hidden layers larger than the cap, keep only the neurons with the
                # highest activation variance -- these are the most differentiated /
                # interesting ones to look at, rather than an arbitrary truncation.
                displayed_indices = [np.array([0])]  # placeholder for input layer, filled below
                displayed_indices[0] = np.arange(layer_sizes[0])  # show all input features

                for hidden_i in range(n_hidden_layers):
                    size = layer_sizes[hidden_i + 1]
                    if size > max_neurons_per_layer:
                        variances = overall_stats[hidden_i][1]  # std per neuron
                        top_idx = np.argsort(variances)[-max_neurons_per_layer:]
                        displayed_indices.append(np.sort(top_idx))
                    else:
                        displayed_indices.append(np.arange(size))

                displayed_indices.append(np.arange(layer_sizes[-1]))  # show all output classes

                fig = go.Figure()
                x_gap = 1.0
                node_positions = []  # node_positions[layer][position_in_display] = (x, y)

                for layer_idx, idx_array in enumerate(displayed_indices):
                    n_shown = len(idx_array)
                    ys = np.linspace(0, 1, n_shown) if n_shown > 1 else np.array([0.5])
                    node_positions.append([(layer_idx * x_gap, y) for y in ys])

                # Draw pruned, weight-sign-colored connections -- only among displayed neurons
                for layer_idx in range(len(layer_sizes) - 1):
                    W = mlp_model.coefs_[layer_idx]  # shape (n_from, n_to), real trained weights
                    from_display = displayed_indices[layer_idx]
                    to_display = displayed_indices[layer_idx + 1]

                    pos_x, pos_y = [], []
                    neg_x, neg_y = [], []

                    for to_pos, to_idx in enumerate(to_display):
                        weights = W[from_display, to_idx]  # weights from displayed sources only
                        keep = np.argsort(np.abs(weights))[-max_connections_per_neuron:]
                        for from_pos in keep:
                            x0, y0 = node_positions[layer_idx][from_pos]
                            x1, y1 = node_positions[layer_idx + 1][to_pos]
                            if weights[from_pos] >= 0:
                                pos_x += [x0, x1, None]
                                pos_y += [y0, y1, None]
                            else:
                                neg_x += [x0, x1, None]
                                neg_y += [y0, y1, None]

                    fig.add_trace(go.Scatter(
                        x=pos_x, y=pos_y, mode="lines",
                        line=dict(color="rgba(70,130,220,0.35)", width=0.7),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig.add_trace(go.Scatter(
                        x=neg_x, y=neg_y, mode="lines",
                        line=dict(color="rgba(220,70,70,0.35)", width=0.7),
                        hoverinfo="skip", showlegend=False,
                    ))

                # Draw nodes -- hidden layers colored by mean activation magnitude, output = fixed color
                for layer_idx, positions in enumerate(node_positions):
                    xs = [p[0] for p in positions]
                    ys = [p[1] for p in positions]
                    idx_array = displayed_indices[layer_idx]

                    if 0 < layer_idx < len(node_positions) - 1:
                        hidden_i = layer_idx - 1
                        mean_act = overall_stats[hidden_i][0][idx_array]
                        labels = [f"Layer {hidden_i+1} · Neuron {real_n}<br>mean activation: {mean_act[i]:.3f}"
                                  for i, real_n in enumerate(idx_array)]
                        customdata = [[hidden_i, int(real_n)] for real_n in idx_array]
                        fig.add_trace(go.Scatter(
                            x=xs, y=ys, mode="markers", name=f"hidden {hidden_i+1}",
                            marker=dict(size=11, color=mean_act, colorscale="Viridis",
                                        showscale=False, line=dict(width=1, color="rgba(0,0,0,0.4)")),
                            text=labels, hoverinfo="text", customdata=customdata,
                        ))
                    elif layer_idx == len(node_positions) - 1:
                        labels = [f"Output: {c}" for c in output_classes]
                        fig.add_trace(go.Scatter(
                            x=xs, y=ys, mode="markers+text", name="output",
                            marker=dict(size=14, color="#e67e22"),
                            text=output_classes, textposition="middle right",
                            hovertext=labels, hoverinfo="text",
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=xs, y=ys, mode="markers", name="input",
                            marker=dict(size=7, color="#8899bb"),
                            hoverinfo="skip",
                        ))

                if any(layer_sizes[i + 1] > max_neurons_per_layer for i in range(n_hidden_layers)):
                    st.caption(
                        "ℹ️ Some hidden layers exceed the slider limit — showing only the "
                        "most differentiated neurons (highest activation variance)."
                    )

                fig.update_layout(
                    showlegend=False, height=700,
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    margin=dict(l=10, r=80, t=10, b=10),
                    clickmode="event+select",
                )

                selection = st.plotly_chart(
                    fig, width='stretch', key="nn_topology", on_select="rerun"
                )

            with col_inspector:
                st.subheader("Neuron Inspector")

                selected_layer, selected_neuron = None, None
                if selection and selection.get("selection", {}).get("points"):
                    point = selection["selection"]["points"][0]
                    cd = point.get("customdata")
                    if cd:
                        selected_layer, selected_neuron = int(cd[0]), int(cd[1])

                if selected_layer is None:
                    st.info("Click a hidden-layer neuron in the diagram to inspect it.")
                else:
                    class_means = hidden_layer_class_means[selected_layer][:, selected_neuron]
                    mu, sigma = overall_stats[selected_layer][0][selected_neuron], overall_stats[selected_layer][1][selected_neuron]

                    best_idx = int(np.argmax(class_means))
                    best_class = class_names[best_idx]
                    other_mean = np.mean(np.delete(class_means, best_idx))
                    selectivity = (class_means[best_idx] - other_mean) / (sigma + 1e-8)
                    selectivity = float(np.clip(selectivity, 0, 1))

                    st.markdown(f"**Layer {selected_layer + 1} · Neuron {selected_neuron}**")
                    st.caption(f"μ={mu:.4f}  σ={sigma:.4f}  selectivity={selectivity:.4f}")
                    st.markdown(f"Fires most for **{best_class}**")

                    st.markdown("**Mean activation per class**")
                    chart_df = pd.DataFrame({"Class": class_names, "Mean Activation": class_means})
                    st.bar_chart(chart_df.set_index("Class"))

            st.divider()
            st.subheader("Attack Simulator (Synthetic Traffic Model)")
            st.caption(
                "Generates real synthetic samples from a saved **per-class CTGAN generator** "
                "(not uniform random noise) and checks how many the MLP correctly detects."
            )

            _, label_to_id_nn, _ = load_label_mapping()
            generators = discover_ctgan_class_generators()
            safe_map = safe_name_to_class_map(label_to_id_nn) if label_to_id_nn else {}

            if not generators:
                st.info("No saved CTGAN generator models found under `CTGAN/Models/`.")
            else:
                display_names = sorted(safe_map.get(g, g) for g in generators.keys())
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    chosen_class = st.selectbox("— select traffic type —", display_names)
                with col_b:
                    n_samples = st.number_input("Count", min_value=1, max_value=1000, value=100)
                with col_c:
                    st.write("")
                    st.write("")
                    run_sim = st.button("Generate & Detect", type="primary")

                if run_sim:
                    safe_name = next(k for k, v in safe_map.items() if v == chosen_class)
                    synthetic = generate_synthetic_class_sample(safe_name, int(n_samples))

                    selected_features = load_selected_features()
                    X_sim = synthetic[[c for c in selected_features if c in synthetic.columns]]
                    X_sim_scaled = mlp_scaler.transform(X_sim)

                    pred_encoded = mlp_model.predict(X_sim_scaled)
                    pred_labels = [id_to_label_nn.get(p, str(p)) for p in pred_encoded]

                    detected = sum(1 for p in pred_labels if p == chosen_class)
                    detection_rate = detected / len(pred_labels) * 100

                    st.metric(
                        f"MLP Detection Rate for '{chosen_class}'",
                        f"{detection_rate:.1f}%",
                        help=f"{detected} of {len(pred_labels)} generated samples correctly detected",
                    )

                    pred_counts = pd.Series(pred_labels).value_counts()
                    st.bar_chart(pred_counts)