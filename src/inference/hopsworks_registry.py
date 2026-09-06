"""Pearls AQI Predictor - Hopsworks Model Registry Connector.

Manages registration and retrieval of the complete EXP-019 production inference bundle
(hybrid model, scaler, canonical schema, empirical residual intervals, and SHA256 manifest)
with explicit version targeting and strict checksum validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_CANDIDATE_MODEL_NAME,
    HOPSWORKS_HOST,
    HOPSWORKS_MODEL_NAME,
    HOPSWORKS_MODEL_VERSION,
    HOPSWORKS_PROJECT,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from src.exceptions import FeatureStoreError, ValidationError
from src.logger import logger

BUNDLE_FILES = [
    "production_hybrid_model.joblib",
    "feature_scaler_v2_weather.joblib",
    "feature_schema_v2_weather.json",
    "empirical_error_intervals.json",
]


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hexadecimal digest of a file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class HopsworksModelRegistryConnector:
    """Connector for registering and fetching full production model bundles with SHA256 verification."""

    def __init__(
        self,
        api_key: str | None = None,
        project_name: str | None = None,
        host: str | None = None,
        model_name: str = HOPSWORKS_MODEL_NAME,
        model_version: int = HOPSWORKS_MODEL_VERSION,
    ) -> None:
        """Initialize Model Registry connector.

        Args:
            api_key: Hopsworks API key.
            project_name: Hopsworks project name.
            host: Hopsworks host URL.
            model_name: Production champion model name in registry.
            model_version: Production champion explicit version integer.
        """
        self.api_key = api_key if api_key is not None else HOPSWORKS_API_KEY
        self.project_name = project_name or HOPSWORKS_PROJECT
        self.host = host or HOPSWORKS_HOST
        self.model_name = model_name
        self.model_version = model_version

        self._project = None
        self._mr = None

    def is_cloud_configured(self) -> bool:
        """Check if Hopsworks cloud credentials are available."""
        return bool(self.api_key and self.api_key.strip() != "")

    def _login(self):
        """Lazy authentication to Hopsworks Model Registry."""
        if self._project is None:
            if not self.is_cloud_configured():
                raise FeatureStoreError("Hopsworks API key is not configured.")
            try:
                import hopsworks
                self._project = hopsworks.login(
                    api_key_value=self.api_key,
                    project=self.project_name,
                    host=self.host,
                )
                self._mr = self._project.get_model_registry()
                logger.info(f"Connected to Hopsworks Model Registry ({self.project_name})")
            except Exception as e:
                logger.warning(f"Hopsworks Model Registry login failed: {e}")
                raise FeatureStoreError(f"Model Registry connection failed: {e}") from e
        return self._mr

    def create_bundle_manifest(self, bundle_dir: Path) -> dict[str, Any]:
        """Generate manifest.json containing metadata and SHA256 checksums for all bundle artifacts.

        Args:
            bundle_dir: Directory containing bundle artifacts.

        Returns:
            Dictionary representation of manifest.
        """
        file_hashes = {}
        for fname in BUNDLE_FILES:
            fpath = bundle_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Required bundle artifact missing: {fpath}")
            file_hashes[fname] = compute_file_sha256(fpath)

        manifest = {
            "model_id": "EXP-019",
            "model_name": self.model_name,
            "model_version": "1.0-production",
            "architecture": "PersistenceAwareHybridModel",
            "canonical_feature_count": 114,
            "output_horizons": 72,
            "feature_schema_version": "v2_weather_enriched",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmarks": {
                "overall_rmse": 75.91,
                "overall_mae": 47.96,
                "overall_r2": 0.380,
                "h1_rmse": 18.06,
                "h72_rmse": 91.13,
            },
            "files": file_hashes,
        }

        manifest_path = bundle_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def verify_bundle_integrity(self, bundle_dir: Path) -> dict[str, Any]:
        """Verify that all files in bundle match their manifest SHA256 checksums (fails closed).

        Args:
            bundle_dir: Path to model bundle directory.

        Returns:
            Verified manifest dictionary.

        Raises:
            ValidationError: If any file is missing or has a mismatched checksum.
        """
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValidationError(f"Integrity violation: manifest.json missing in {bundle_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        files = manifest.get("files", {})
        for fname, expected_hash in files.items():
            fpath = bundle_dir / fname
            if not fpath.exists():
                raise ValidationError(f"Integrity violation: artifact {fname} missing from bundle {bundle_dir}")
            actual_hash = compute_file_sha256(fpath)
            if actual_hash != expected_hash:
                raise ValidationError(
                    f"Integrity violation: SHA256 checksum mismatch for {fname} "
                    f"(expected {expected_hash[:12]}..., got {actual_hash[:12]}...). "
                    "Refusing to load corrupted model artifacts."
                )

        logger.info(f"Model bundle integrity verified successfully for {bundle_dir}")
        return manifest

    def build_local_bundle(self, output_dir: Path) -> Path:
        """Assemble the complete production bundle from local source paths into a unified directory.

        Args:
            output_dir: Destination bundle directory.

        Returns:
            Path to assembled bundle directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Source locations
        sources = {
            "production_hybrid_model.joblib": MODELS_DIR / "production_hybrid_model.joblib",
            "feature_scaler_v2_weather.joblib": MODELS_DIR / "feature_scaler_v2_weather.joblib",
            "feature_schema_v2_weather.json": PROCESSED_DATA_DIR / "feature_schema_v2_weather.json",
            "empirical_error_intervals.json": MODELS_DIR / "walk_forward" / "empirical_error_intervals.json",
        }

        for fname, src_path in sources.items():
            if not src_path.exists():
                raise FileNotFoundError(f"Source artifact missing: {src_path}")
            shutil.copy2(src_path, output_dir / fname)

        self.create_bundle_manifest(output_dir)
        return output_dir

    def register_production_champion(self, bundle_dir: Path, description: str | None = None) -> Any:
        """Register the complete verified EXP-019 bundle in Hopsworks Model Registry.

        Args:
            bundle_dir: Directory containing bundle artifacts and manifest.json.
            description: Optional model description.

        Returns:
            Registered model object.
        """
        manifest = self.verify_bundle_integrity(bundle_dir)
        mr = self._login()

        if description is None:
            description = (
                "EXP-019 Production Champion: PersistenceAwareHybridModel "
                "with 114 canonical features predicting 72-hour continuous AQI."
            )

        logger.info(f"Registering production champion bundle under name '{self.model_name}'...")
        model = mr.python.create_model(
            name=self.model_name,
            metrics=manifest["benchmarks"],
            description=description,
            input_example=[0.0] * 114,
        )
        model.save(str(bundle_dir))
        logger.info(f"Successfully registered model '{self.model_name}' in Hopsworks Model Registry")
        return model

    def fetch_production_champion(self, download_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
        """Fetch production champion from Hopsworks by explicit name and version with checksum validation.

        Args:
            download_dir: Directory to save downloaded model bundle.

        Returns:
            Tuple of (bundle_path, manifest_dict).
        """
        if self.is_cloud_configured():
            try:
                mr = self._login()
                # Explicit lookup by name and version - never query 'latest' or 'best'
                model = mr.get_model(name=self.model_name, version=self.model_version)
                if download_dir is None:
                    download_dir = MODELS_DIR / "hopsworks_bundle"
                download_dir.mkdir(parents=True, exist_ok=True)

                downloaded_path = Path(model.download(str(download_dir)))
                manifest = self.verify_bundle_integrity(downloaded_path)
                return downloaded_path, manifest
            except ValidationError:
                # Corrupt checksum or missing artifact must fail closed
                raise
            except Exception as e:
                logger.warning(f"Cloud Model Registry fetch failed ({e}); activating local model fallback.")

        # Local fallback
        local_bundle_dir = MODELS_DIR / "bundle"
        if not (local_bundle_dir / "manifest.json").exists():
            self.build_local_bundle(local_bundle_dir)

        manifest = self.verify_bundle_integrity(local_bundle_dir)
        return local_bundle_dir, manifest

    def register_candidate_model(self, candidate_dir: Path, metrics: dict[str, Any]) -> Any:
        """Register candidate model in isolated candidate namespace without shadowing champion."""
        mr = self._login()
        candidate_model = mr.python.create_model(
            name=HOPSWORKS_CANDIDATE_MODEL_NAME,
            metrics=metrics,
            description="Automated candidate model evaluation artifact.",
        )
        candidate_model.save(str(candidate_dir))
        return candidate_model
