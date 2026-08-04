#!/usr/bin/env bash
# Build step for the deployed service. Render runs this from webapp_django/.
set -o errexit

pip install -r requirements.txt

# The model .pkl files live in Git LFS. A checkout without LFS leaves 130-byte
# pointer files behind, and the app would then die at the first prediction with
# a confusing UnpicklingError. Pull them, then verify -- failing the build here
# is far easier to debug than a 503 in production.
MODELS_DIR="../webapp_data/Results/Models"

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install
  git lfs pull --include="webapp_data/Results/Models/*.pkl,webapp_data/Processed_Data/selected_features.pkl" || true
else
  echo "WARNING: git-lfs not available in the build image."
fi

REQUIRED="HistGradientBoosting_Tuned.pkl XGBoost_Tuned.pkl MLP_Tuned.pkl MLP_StandardScaler.pkl"
for f in $REQUIRED; do
  path="$MODELS_DIR/$f"
  if [ ! -f "$path" ]; then
    echo "BUILD FAILED: $f is missing from $MODELS_DIR"
    exit 1
  fi
  if head -c 40 "$path" | grep -q "git-lfs.github.com"; then
    echo "BUILD FAILED: $f is still a Git LFS pointer, not a real model."
    echo "  The build image could not fetch LFS objects. See DEPLOY.md for the"
    echo "  fallback: commit the four runtime models as ordinary git objects."
    exit 1
  fi
done
echo "All 4 model files present and valid."

python manage.py collectstatic --noinput
python manage.py migrate
