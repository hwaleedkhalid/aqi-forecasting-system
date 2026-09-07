"""Pearls AQI Predictor - Runtime Asset Resolver & Integrity Validator.

Resolves versioned runtime deployment assets (production bundle, explainability artifacts,
and bootstrap feature vector) with deterministic modes (runtime, legacy, auto) and
fail-closed integrity checks.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from src.exceptions import ValidationError
from src.logger import logger

# Canonical Runtime Paths
RUNTIME_DIR = DATA_DIR / "runtime"
RUNTIME_PRODUCTION_DIR = RUNTIME_DIR / "production"
RUNTIME_EXPLAINABILITY_DIR = RUNTIME_DIR / "explainability"
RUNTIME_BOOTSTRAP_DIR = RUNTIME_DIR / "bootstrap"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hexadecimal digest of a file (normalizing JSON text to LF)."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # For JSON text files, normalize CRLF to LF to guarantee cross-OS invariance
    if file_path.suffix.lower() == ".json":
        text = file_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()



class RuntimeAssetResolver:
    """Discovers, validates, and provides deterministic paths to production runtime assets."""

    def __init__(
        self,
        mode: str | None = None,
        base_runtime_dir: Path | str | None = None,
    ) -> None:
        """Initialize resolver.

        Args:
            mode: Asset resolution mode ('runtime', 'legacy', or 'auto').
                  Defaults to RUNTIME_ASSET_MODE env var or 'auto'.
            base_runtime_dir: Optional override for base runtime directory (default: data/runtime).
        """
        self.mode = (mode or os.environ.get("RUNTIME_ASSET_MODE", "auto")).lower()
        if self.mode not in {"runtime", "legacy", "auto"}:
            raise ValidationError(f"Invalid RUNTIME_ASSET_MODE '{self.mode}': must be 'runtime', 'legacy', or 'auto'")

        self.runtime_dir = Path(base_runtime_dir or RUNTIME_DIR)
        self.production_dir = self.runtime_dir / "production"
        self.explainability_dir = self.runtime_dir / "explainability"
        self.bootstrap_dir = self.runtime_dir / "bootstrap"

    def get_model_path(self) -> Path:
        """Resolve production hybrid model joblib path based on active mode."""
        runtime_path = self.production_dir / "production_hybrid_model.joblib"
        legacy_path = MODELS_DIR / "production_hybrid_model.joblib"

        if self.mode == "runtime":
            return runtime_path
        elif self.mode == "legacy":
            return legacy_path
        else:  # auto
            return runtime_path if runtime_path.exists() else legacy_path

    def get_scaler_path(self) -> Path:
        """Resolve feature scaler joblib path based on active mode."""
        runtime_path = self.production_dir / "feature_scaler_v2_weather.joblib"
        legacy_path = MODELS_DIR / "feature_scaler_v2_weather.joblib"

        if self.mode == "runtime":
            return runtime_path
        elif self.mode == "legacy":
            return legacy_path
        else:
            return runtime_path if runtime_path.exists() else legacy_path

    def get_schema_path(self) -> Path:
        """Resolve canonical 114-feature schema JSON path based on active mode."""
        runtime_path = self.production_dir / "feature_schema_v2_weather.json"
        legacy_path = PROCESSED_DATA_DIR / "feature_schema_v2_weather.json"

        if self.mode == "runtime":
            return runtime_path
        elif self.mode == "legacy":
            return legacy_path
        else:
            return runtime_path if runtime_path.exists() else legacy_path

    def get_error_intervals_path(self) -> Path:
        """Resolve empirical residual error intervals JSON path based on active mode."""
        runtime_path = self.production_dir / "empirical_error_intervals.json"
        legacy_path = MODELS_DIR / "walk_forward" / "empirical_error_intervals.json"

        if self.mode == "runtime":
            return runtime_path
        elif self.mode == "legacy":
            return legacy_path
        else:
            return runtime_path if runtime_path.exists() else legacy_path

    def get_production_manifest_path(self) -> Path:
        """Resolve production bundle manifest JSON path."""
        runtime_path = self.production_dir / "manifest.json"
        legacy_path = MODELS_DIR / "bundle" / "manifest.json"

        if self.mode == "runtime":
            return runtime_path
        elif self.mode == "legacy":
            return legacy_path
        else:
            return runtime_path if runtime_path.exists() else legacy_path

    def get_explainability_dir(self) -> Path:
        """Resolve explainability directory containing background and global importance."""
        if self.mode == "runtime":
            return self.explainability_dir
        elif self.mode == "legacy":
            return MODELS_DIR / "explainability"
        else:
            return self.explainability_dir if self.explainability_dir.exists() else MODELS_DIR / "explainability"

    def get_bootstrap_feature_path(self) -> Path:
        """Resolve bootstrap feature vector JSON path."""
        return self.bootstrap_dir / "latest_feature_vector.json"

    def verify_runtime_integrity(self) -> dict[str, Any]:
        """Verify checksums of runtime production bundle against manifest (fails closed)."""
        manifest_path = self.get_production_manifest_path()
        if not manifest_path.exists():
            raise ValidationError(f"Runtime manifest missing at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        files = manifest.get("files", {})
        bundle_dir = manifest_path.parent
        for fname, expected_hash in files.items():
            fpath = bundle_dir / fname
            if not fpath.exists():
                raise ValidationError(f"Runtime artifact missing: {fname} at {fpath}")
            actual_hash = compute_file_sha256(fpath)
            if actual_hash != expected_hash:
                raise ValidationError(
                    f"Integrity violation on {fname}: expected SHA256 {expected_hash[:12]}..., "
                    f"got {actual_hash[:12]}.... Refusing to load corrupted model."
                )

        schema_path = self.get_schema_path()
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        features = schema.get("feature_names", schema.get("features", []))
        if len(features) != 114:
            raise ValidationError(f"Expected exactly 114 canonical features in runtime schema, got {len(features)}")

        logger.info(f"Production runtime bundle integrity verified ({len(files)} files checked).")
        return manifest

    def verify_explainability_integrity(self) -> dict[str, Any]:
        """Verify explainability manifest and binding to production model SHA256 (fails closed)."""
        expl_dir = self.get_explainability_dir()
        expl_manifest_path = expl_dir / "explainer_manifest.json"
        if not expl_manifest_path.exists():
            raise ValidationError(f"Explainability manifest missing at {expl_manifest_path}")

        with open(expl_manifest_path, "r", encoding="utf-8") as f:
            expl_manifest = json.load(f)

        # Verify model SHA256 binding
        model_path = self.get_model_path()
        if model_path.exists():
            current_model_hash = compute_file_sha256(model_path)
            bound_model_hash = expl_manifest.get("production_model_sha256")
            if bound_model_hash and current_model_hash != bound_model_hash:
                raise ValidationError(
                    f"Explainability artifact mismatch: bound model hash {bound_model_hash[:12]}... "
                    f"does not match current model {current_model_hash[:12]}..."
                )

        # Verify background file hash
        bg_path = expl_dir / "shap_background.npy"
        if bg_path.exists() and "shap_background_sha256" in expl_manifest:
            actual_bg_hash = compute_file_sha256(bg_path)
            if actual_bg_hash != expl_manifest["shap_background_sha256"]:
                raise ValidationError("Integrity violation on shap_background.npy: SHA256 mismatch")

        return expl_manifest

    def load_bootstrap_feature_vector(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load and validate the committed canonical bootstrap feature vector with strict schema checks."""
        bootstrap_path = self.get_bootstrap_feature_path()
        if not bootstrap_path.exists():
            raise FileNotFoundError(f"Bootstrap feature vector not found at {bootstrap_path}")

        with open(bootstrap_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load canonical schema and check hash binding
        schema_path = self.get_schema_path()
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        current_schema_hash = compute_file_sha256(schema_path)
        bound_schema_hash = data.get("schema_sha256")
        if bound_schema_hash and current_schema_hash != bound_schema_hash:
            raise ValidationError(
                f"Bootstrap schema hash mismatch: expected {bound_schema_hash[:12]}..., got {current_schema_hash[:12]}..."
            )

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
        canonical_features = schema_data.get("feature_names", [])

        feature_dict = data.get("features", {})
        if len(feature_dict) != 114:
            raise ValidationError(f"Bootstrap features count mismatch: expected 114, found {len(feature_dict)}")

        # Verify all canonical features are present
        missing = [col for col in canonical_features if col not in feature_dict]
        if missing:
            raise ValidationError(f"Bootstrap feature vector missing canonical columns: {missing[:5]}")

        # Construct single-row DataFrame in exact canonical order and validate finite numeric values
        ordered_values = []
        for col in canonical_features:
            val = feature_dict[col]
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValidationError(f"Invalid non-finite numeric value for feature {col}: {val}")
            ordered_values.append(float(val))

        df = pd.DataFrame([ordered_values], columns=canonical_features)

        # Timestamp validation
        obs_raw = data.get("input_observed_at")
        if not obs_raw:
            raise ValidationError("Bootstrap feature vector missing 'input_observed_at' timestamp")

        try:
            obs_dt = datetime.fromisoformat(obs_raw.replace("Z", "+00:00"))
            if obs_dt.tzinfo is None:
                obs_dt = obs_dt.replace(tzinfo=timezone.utc)
        except Exception as e:
            raise ValidationError(f"Invalid timestamp in bootstrap vector '{obs_raw}': {e}") from e

        metadata = {
            "source": "bootstrap",
            "input_observed_at": obs_dt,
            "current_aqi": float(data.get("current_aqi", ordered_values[9])),
            "dominant_pollutant": str(data.get("dominant_pollutant", "pm2_5")),
            "schema_version": data.get("schema_version", "v2_weather_enriched"),
            "pollutants": data.get("pollutants", {}),
            "weather": data.get("weather", {}),
        }

        return df, metadata
