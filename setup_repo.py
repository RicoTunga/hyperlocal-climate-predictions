"""Scaffold the agro-forecast repo. Run from the repo root: python setup_repo.py
Existing files (e.g. README.md) are never overwritten."""
from pathlib import Path

ROOT = Path(".")

DIRS = [
    "config",
    "data/raw", "data/processed",
    "notebooks",
    "src/ingestion", "src/features", "src/models", "src/explain",
    "models",
    "app",
    "tests",
]

# Folders that are Python packages (importable from notebooks and the app)
PACKAGES = ["src", "src/ingestion", "src/features", "src/models", "src/explain"]

# Folders that must exist in git even though their contents are ignored
KEEP = ["data/raw", "data/processed", "models", "notebooks", "app", "tests"]

FILES = {
    "requirements.txt": """\
# Core data + ML
pandas
numpy
pyarrow
scikit-learn
xgboost
lightgbm
catboost
optuna
shap
matplotlib

# Data acquisition
xarray
netCDF4
cdsapi
earthengine-api
requests
pyyaml
python-dotenv

# Notebooks
ipykernel
jupyter

# Added later (dashboard phase)
# streamlit
# streamlit-authenticator
# psycopg2-binary
""",
    ".gitignore": """\
# Data and artefacts (re-create with the ingestion scripts)
data/raw/*
data/processed/*
models/*
!**/.gitkeep

# Secrets - never commit
.env
*.json.key
credentials*.json
.cdsapirc

# Python / Jupyter
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
venv/
.pytest_cache/

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/
""",
    ".env.example": """\
# Copy to .env and fill in. .env is git-ignored.
# CDS API credentials normally live in ~/.cdsapirc (outside the repo).
GEE_PROJECT=your-google-cloud-project-id
TAHMO_API_USER=
TAHMO_API_PASSWORD=
""",
    "config/sites.yaml": """\
# One entry per study site. Code should read sites from here, never hard-code coordinates.
# Bounding box order: lon_min, lon_max, lat_min, lat_max
period:
  start: 2016-01-01
  end: 2025-12-31

sites:
  nakuru:
    name: Nakuru County (pilot)
    bbox: {lon_min: 35.50, lon_max: 36.65, lat_min: -1.10, lat_max: 0.30}
    points:                      # representative grid points (fill in after exploring data)
      - {name: nakuru_centre, lat: -0.30, lon: 36.07}
  busia:
    name: Busia (station data from Mr Solomon)
    bbox: {lon_min: 33.90, lon_max: 34.30, lat_min: 0.25, lat_max: 0.65}   # PLACEHOLDER: replace with real coordinates
    points:
      - {name: busia_town, lat: 0.46, lon: 34.11}                          # PLACEHOLDER

forecast:
  horizons: [1, 2, 3, 4, 5, 6, 7]
  targets: [precip_mm, rh_pct]
""",
    "src/ingestion/README.md": "Fetchers for CHIRPS (Earth Engine), ERA5 (CDS API), NASA POWER, TAHMO / Busia station data.\n",
    "src/features/README.md": "Lag, rolling-window and cyclic calendar feature builders.\n",
    "src/models/README.md": "Baselines, training, TimeSeriesSplit evaluation, Optuna tuning.\n",
    "src/explain/README.md": "SHAP TreeExplainer wrappers and plots.\n",
}


def write_if_missing(path: Path, content: str = "") -> None:
    if path.exists():
        print(f"skip    {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"created {path}")


def main() -> None:
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    for p in PACKAGES:
        write_if_missing(ROOT / p / "__init__.py")
    for k in KEEP:
        write_if_missing(ROOT / k / ".gitkeep")
    for name, content in FILES.items():
        write_if_missing(ROOT / name, content)
    print("\nDone. Next: create a venv, pip install -r requirements.txt, then git add/commit/push.")


if __name__ == "__main__":
    main()
