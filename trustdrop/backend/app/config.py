"""Central configuration. Values can be overridden with TRUSTDROP_* env vars."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]  # .../trustdrop


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRUSTDROP_")

    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"
    reports_dir: Path = ROOT_DIR / "reports"
    # Real UCI file. Download from https://archive.ics.uci.edu/dataset/697 and save here.
    dataset_filename: str = "data.csv"
    # Synthetic data is ONLY for checking that the code runs; never for reported results.
    synthetic_filename: str = "synthetic_dev.csv"
    database_url: str = f"sqlite:///{ROOT_DIR / 'data' / 'trustdrop.db'}"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    n_perturbations: int = 30
    random_seed: int = 42
    test_size: float = 0.20


settings = Settings()
