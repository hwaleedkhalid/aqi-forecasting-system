"""Pearls AQI Predictor - Systematic Experiment Registry.

Tracks, serializes, and compares all model training and tuning experiments
evaluated strictly on chronological training-only validation folds, ensuring
reproducibility and preventing test-set leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import pandas as pd

from src.config import MODELS_DIR
from src.exceptions import ValidationError
from src.logger import logger

EXPERIMENTS_DIR = MODELS_DIR / "experiments"


@dataclass
class ExperimentRecord:
    """Dataclass storing comprehensive metadata and validation metrics for a single experiment."""

    experiment_id: str
    feature_version: str
    model_name: str
    hyperparameters: dict[str, Any]
    strategy: str
    cv_scheme: str
    val_rmse_mean: float
    val_rmse_std: float
    val_mae_mean: float
    val_r2_mean: float
    val_h1_rmse: float
    val_h24_rmse: float
    val_h72_rmse: float
    training_time_sec: float
    notes: str = ""
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        """Construct record from dictionary."""
        return cls(**data)


class ExperimentRegistry:
    """Manages experiment tracking, persistence, and leaderboard generation."""

    def __init__(self, registry_dir: Path = EXPERIMENTS_DIR) -> None:
        """Initialize ExperimentRegistry.

        Args:
            registry_dir: Directory where experiment JSON records and registry tables are saved.
        """
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: dict[str, ExperimentRecord] = {}
        self.load_registry()

    def log_experiment(self, record: ExperimentRecord) -> str:
        """Log and persist a new experiment run.

        Args:
            record: ExperimentRecord instance.

        Returns:
            experiment_id string.
        """
        self.experiments[record.experiment_id] = record

        # Save individual experiment record JSON
        exp_file = self.registry_dir / f"{record.experiment_id}.json"
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        # Update centralized registry files
        self.save_registry()
        logger.info(
            f"Registered experiment {record.experiment_id} ({record.model_name} | {record.strategy}): "
            f"Val RMSE={record.val_rmse_mean:.2f} (+/-{record.val_rmse_std:.2f}), "
            f"Val MAE={record.val_mae_mean:.2f}, Val R²={record.val_r2_mean:.4f}"
        )
        return record.experiment_id

    def get_leaderboard(self) -> pd.DataFrame:
        """Generate a sorted summary DataFrame of all logged experiments.

        Returns:
            DataFrame sorted by val_rmse_mean in ascending order.
        """
        if not self.experiments:
            return pd.DataFrame()

        rows = []
        for exp in self.experiments.values():
            rows.append({
                "Experiment ID": exp.experiment_id,
                "Model": exp.model_name,
                "Feature Version": exp.feature_version,
                "Strategy": exp.strategy,
                "Val RMSE (Mean)": round(exp.val_rmse_mean, 2),
                "Val RMSE (Std)": round(exp.val_rmse_std, 2),
                "Val MAE": round(exp.val_mae_mean, 2),
                "Val R²": round(exp.val_r2_mean, 4),
                "h+1 RMSE": round(exp.val_h1_rmse, 2),
                "h+24 RMSE": round(exp.val_h24_rmse, 2),
                "h+72 RMSE": round(exp.val_h72_rmse, 2),
                "Train Time (s)": round(exp.training_time_sec, 2),
                "Notes": exp.notes,
            })

        df_leaderboard = pd.DataFrame(rows).sort_values("Val RMSE (Mean)").reset_index(drop=True)
        return df_leaderboard

    def get_best_experiment(self, metric: str = "val_rmse_mean") -> ExperimentRecord | None:
        """Retrieve the best experiment record based on a specific validation metric.

        Args:
            metric: Metric attribute name (e.g. 'val_rmse_mean' or 'val_r2_mean').

        Returns:
            Winning ExperimentRecord instance or None if registry is empty.
        """
        if not self.experiments:
            return None

        if metric in ["val_rmse_mean", "val_mae_mean", "val_h1_rmse", "val_h24_rmse", "val_h72_rmse"]:
            return min(self.experiments.values(), key=lambda x: getattr(x, metric))
        elif metric == "val_r2_mean":
            return max(self.experiments.values(), key=lambda x: getattr(x, metric))
        else:
            raise ValidationError(f"Unsupported leaderboard sorting metric: '{metric}'")

    def save_registry(self) -> None:
        """Persist registry overview table to registry.json and registry.csv."""
        summary_json_path = self.registry_dir / "registry.json"
        summary_csv_path = self.registry_dir / "registry.csv"

        raw_dict = {k: v.to_dict() for k, v in self.experiments.items()}
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_dict, f, indent=2)

        df_board = self.get_leaderboard()
        if not df_board.empty:
            df_board.to_csv(summary_csv_path, index=False)

    def load_registry(self) -> None:
        """Load all experiment records from registry directory."""
        summary_json_path = self.registry_dir / "registry.json"
        if summary_json_path.exists():
            try:
                with open(summary_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    self.experiments[k] = ExperimentRecord.from_dict(v)
            except Exception as err:
                logger.warning(f"Could not load registry.json: {err}")
