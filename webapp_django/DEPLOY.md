# Deploying the NIDS web app

## Before anything else: the Git LFS problem

The trained models are stored in **Git LFS**. A plain `git clone` on a host that
does not pull LFS gives you 130-byte pointer files instead of models, and the app
raises `UnpicklingError: invalid load key, 'v'` on the first prediction.

Only ~12.6 MB is actually needed at runtime:

| File | Size |
|---|---|
| `HistGradientBoosting_Tuned.pkl` | 4.0 MB |
| `XGBoost_Tuned.pkl` | 6.8 MB |
| `MLP_Tuned.pkl` | 2.4 MB |
| `MLP_StandardScaler.pkl` | 2 KB |

`build.sh` runs `git lfs pull` limited to those files. If your host's build image
has no `git-lfs`, the fallback is to commit those four files as ordinary git
objects (12.6 MB is fine for plain git):

```bash
git rm --cached webapp_data/Results/Models/HistGradientBoosting_Tuned.pkl
printf 'webapp_data/Results/Models/*.pkl -filter -diff -merge text=auto\n' >> .gitattributes
git add -f webapp_data/Results/Models/HistGradientBoosting_Tuned.pkl
```

Note the repo as a whole is **664 MB** (101 MB of it LFS). Some hosts cap build
size — if you hit that, deploy only what the app needs rather than the whole repo.

## Required environment variables

Copy `.env.example` and fill it in. The app **refuses to start** with
`IDS_DEBUG=0` and no `IDS_SECRET_KEY`, which is deliberate — it stops the
throwaway development key reaching production.

```bash
python -c "from django.core.management.utils import get_random_secret_key as k; print(k())"
```

`IDS_SECURE=1` switches on the SSL redirect, HSTS and secure cookies. Turn it on
**only once the host actually terminates TLS** — otherwise every request
redirects to `https://` and the site looks like it is hanging.

## Render (free tier)

1. New → Web Service → connect the GitHub repo
2. **Root directory:** `webapp_django`
3. **Build command:** `./build.sh`
4. **Start command:** `gunicorn ids_web.wsgi:application`
5. Add the environment variables from `.env.example`, setting
   `IDS_ALLOWED_HOSTS` and `IDS_CSRF_TRUSTED_ORIGINS` to the URL Render assigns

The free tier sleeps after inactivity, so the first request after a nap takes
30–60 s — and this app additionally unpickles ~12 MB of models on first
prediction. Expect a slow first hit, then fast ones.

## Database

Works as-is on SQLite, which is the simplest thing that runs. Note that hosts
with ephemeral disks reset SQLite on every deploy, so prediction history will not
survive — fine for a demo, not for anything you need to keep.

For MySQL, set `IDS_DB_ENGINE=mysql` plus the credentials and create the database
first:

```sql
CREATE DATABASE ids_nids CHARACTER SET utf8mb4;
```

## Verifying a deployment

```bash
curl -o /dev/null -w "%{http_code}\n" https://your-app/          # 200
curl -o /dev/null -w "%{http_code}\n" https://your-app/batch/    # 200
```

Then load an example flow on the home page and click **Classify flow**. If the
models failed to load you get a 503 with a message naming the missing file rather
than a generic error.

## Running the production stack locally

```bash
cd webapp_django
export IDS_DEBUG=0
export IDS_SECRET_KEY="$(python -c 'from django.core.management.utils import get_random_secret_key as k; print(k())')"
export IDS_ALLOWED_HOSTS=127.0.0.1,localhost
python manage.py collectstatic --noinput
gunicorn ids_web.wsgi:application --bind 127.0.0.1:8000
```

Leave `IDS_SECURE` unset locally — there is no TLS on `127.0.0.1`.
