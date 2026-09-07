"""Pearls AQI Predictor - Hopsworks Model Registry Connector.

Manages registration and retrieval of the complete EXP-019 production inference bundle
(hybrid model, scaler, canonical schema, empirical residual intervals, and SHA256 manifest)
with explicit version targeting and strict checksum validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
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

EXPLAINABILITY_FILES = [
    "shap_background.npy",
    "global_shap_importance.json",
]


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
            fpath = bundle_dir / "production" / fname
            if not fpath.exists():
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
                "overall_mae": 53.55,
                "overall_r2": 0.4858,
                "h1_rmse": 50.43,
                "h72_rmse": 77.43,
            },
            "files": file_hashes,
        }

        # Write manifest to bundle root
        manifest_path = bundle_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Also write inside production/ subdirectory if present
        prod_dir = bundle_dir / "production"
        if prod_dir.exists():
            with open(prod_dir / "manifest.json", "w", encoding="utf-8") as f:
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
        manifest_path = bundle_dir / "production" / "manifest.json"
        if not manifest_path.exists():
            manifest_path = bundle_dir / "manifest.json"

        if not manifest_path.exists():
            raise ValidationError(f"Integrity violation: manifest.json missing in {bundle_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        files = manifest.get("files", {})
        for fname, expected_hash in files.items():
            candidate_paths = [bundle_dir / fname, bundle_dir / "production" / fname]
            found = False
            for fpath in candidate_paths:
                if fpath.exists():
                    found = True
                    actual_hash = compute_file_sha256(fpath)
                    if actual_hash != expected_hash:
                        raise ValidationError(
                            f"Integrity violation: SHA256 checksum mismatch for {fname} "
                            f"(expected {expected_hash[:12]}..., got {actual_hash[:12]}...). "
                            "Refusing to load corrupted model artifacts."
                        )
            if not found:
                raise ValidationError(f"Integrity violation: artifact {fname} missing from bundle {bundle_dir}")

        # If explainability manifest exists, verify explainability files
        expl_manifest_path = bundle_dir / "explainability" / "explainer_manifest.json"
        if not expl_manifest_path.exists():
            expl_manifest_path = bundle_dir / "explainer_manifest.json"

        if expl_manifest_path.exists():
            with open(expl_manifest_path, "r", encoding="utf-8") as f:
                expl_manifest = json.load(f)
            expl_dir = expl_manifest_path.parent

            bg_path = expl_dir / "shap_background.npy"
            if bg_path.exists() and "shap_background_sha256" in expl_manifest:
                bg_hash = compute_file_sha256(bg_path)
                if bg_hash != expl_manifest["shap_background_sha256"]:
                    raise ValidationError("Integrity violation: shap_background.npy checksum mismatch")

            gi_path = expl_dir / "global_shap_importance.json"
            if gi_path.exists() and "global_shap_importance_sha256" in expl_manifest:
                gi_hash = compute_file_sha256(gi_path)
                if gi_hash != expl_manifest["global_shap_importance_sha256"]:
                    raise ValidationError("Integrity violation: global_shap_importance.json checksum mismatch")

        manifest["feature_count"] = manifest.get("feature_count", manifest.get("canonical_feature_count", 114))
        manifest["canonical_feature_count"] = manifest.get("canonical_feature_count", manifest.get("feature_count", 114))

        logger.info(f"Model bundle integrity verified successfully for {bundle_dir}")
        return manifest

    def build_local_bundle(self, output_dir: Path, include_explainability: bool = True) -> Path:
        """Assemble the complete production bundle into output_dir.

        Creates:
          output_dir/production/ (production model, scaler, schema, intervals, manifest)
          output_dir/explainability/ (shap background, global importance, explainer manifest)
          output_dir/ (flat copies of production artifacts and manifest for flat-bundle consumers)

        Args:
            output_dir: Destination bundle directory.
            include_explainability: Whether to package explainability artifacts.

        Returns:
            Path to assembled bundle directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        prod_dir = output_dir / "production"
        prod_dir.mkdir(parents=True, exist_ok=True)

        runtime_prod_dir = DATA_DIR / "runtime" / "production"
        runtime_expl_dir = DATA_DIR / "runtime" / "explainability"

        sources = {
            "production_hybrid_model.joblib": (
                runtime_prod_dir / "production_hybrid_model.joblib"
                if (runtime_prod_dir / "production_hybrid_model.joblib").exists()
                else MODELS_DIR / "production_hybrid_model.joblib"
            ),
            "feature_scaler_v2_weather.joblib": (
                runtime_prod_dir / "feature_scaler_v2_weather.joblib"
                if (runtime_prod_dir / "feature_scaler_v2_weather.joblib").exists()
                else MODELS_DIR / "feature_scaler_v2_weather.joblib"
            ),
            "feature_schema_v2_weather.json": (
                runtime_prod_dir / "feature_schema_v2_weather.json"
                if (runtime_prod_dir / "feature_schema_v2_weather.json").exists()
                else PROCESSED_DATA_DIR / "feature_schema_v2_weather.json"
            ),
            "empirical_error_intervals.json": (
                runtime_prod_dir / "empirical_error_intervals.json"
                if (runtime_prod_dir / "empirical_error_intervals.json").exists()
                else MODELS_DIR / "walk_forward" / "empirical_error_intervals.json"
            ),
        }

        for fname, src_path in sources.items():
            if not src_path.exists():
                raise FileNotFoundError(f"Source artifact missing: {src_path}")
            shutil.copy2(src_path, prod_dir / fname)
            shutil.copy2(src_path, output_dir / fname)

        src_manifest = runtime_prod_dir / "manifest.json"
        if src_manifest.exists():
            shutil.copy2(src_manifest, prod_dir / "manifest.json")
            shutil.copy2(src_manifest, output_dir / "manifest.json")
        else:
            self.create_bundle_manifest(output_dir)

        if include_explainability:
            expl_dir = output_dir / "explainability"
            expl_dir.mkdir(parents=True, exist_ok=True)
            expl_sources = {
                "shap_background.npy": (
                    runtime_expl_dir / "shap_background.npy"
                    if (runtime_expl_dir / "shap_background.npy").exists()
                    else MODELS_DIR / "explainability" / "shap_background.npy"
                ),
                "global_shap_importance.json": (
                    runtime_expl_dir / "global_shap_importance.json"
                    if (runtime_expl_dir / "global_shap_importance.json").exists()
                    else MODELS_DIR / "explainability" / "global_shap_importance.json"
                ),
                "explainer_manifest.json": (
                    runtime_expl_dir / "explainer_manifest.json"
                    if (runtime_expl_dir / "explainer_manifest.json").exists()
                    else MODELS_DIR / "explainability" / "explainer_manifest.json"
                ),
            }
            for fname, src_path in expl_sources.items():
                if src_path.exists():
                    shutil.copy2(src_path, expl_dir / fname)

        self.verify_bundle_integrity(output_dir)
        return output_dir

    def register_production_champion(self, bundle_dir: Path, description: str | None = None) -> Any:
        """Register the complete verified EXP-019 bundle in Hopsworks Model Registry with safe idempotency.

        Args:
            bundle_dir: Directory containing bundle artifacts and manifest.json.
            description: Optional model description.

        Returns:
            Registered model object.

        Raises:
            ValidationError: If existing version 1 has mismatched checksums (fails closed).
        """
        manifest = self.verify_bundle_integrity(bundle_dir)
        mr = self._login()

        existing_model = None
        try:
            existing_model = mr.get_model(name=self.model_name, version=self.model_version)
        except Exception as e:
            logger.debug(f"Model lookup returned exception (not found): {e}")
            existing_model = None

        if existing_model is not None:
            logger.info(
                f"Model '{self.model_name}' version {self.model_version} already exists in registry. "
                "Verifying checksum parity to guarantee safe idempotency..."
            )
            with tempfile.TemporaryDirectory() as td:
                temp_dl = Path(existing_model.download(td))
                try:
                    downloaded_manifest = self.verify_bundle_integrity(temp_dl)
                except ValidationError as ve:
                    raise ValidationError(
                        f"Existing registered model '{self.model_name}' v{self.model_version} corrupted: {ve}"
                    ) from ve

                for fname, expected_hash in manifest["files"].items():
                    downloaded_hash = downloaded_manifest["files"].get(fname)
                    if downloaded_hash != expected_hash:
                        raise ValidationError(
                            f"Conflict: Registered model version {self.model_version} checksum mismatch on {fname}: "
                            f"registered={downloaded_hash}, local={expected_hash}. "
                            "Refusing to overwrite existing version."
                        )

            logger.info(
                f"Model '{self.model_name}' version {self.model_version} is already registered with identical checksums. "
                "Safe re-run idempotency verified. Returning existing model."
            )
            return existing_model

        schema_file = bundle_dir / "production" / "feature_schema_v2_weather.json"
        if not schema_file.exists():
            schema_file = bundle_dir / "feature_schema_v2_weather.json"
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
        canonical_features = schema_data.get("feature_names", [])

        input_df = pd.DataFrame([[0.0] * len(canonical_features)], columns=canonical_features)
        output_df = pd.DataFrame([[0.0] * 72], columns=[f"h{i}" for i in range(1, 73)])

        model_schema = None
        try:
            from hsml.schema import Schema
            from hsml.model_schema import ModelSchema
            model_schema = ModelSchema(input_schema=Schema(input_df), output_schema=Schema(output_df))
        except Exception as e:
            logger.warning(f"Could not construct ModelSchema: {e}")

        if description is None:
            description = (
                "EXP-019 validated production champion: PersistenceAwareHybridModel "
                "(h1-6 LightGBM, h7-37 Ridge, h38-72 Ridge + persistence blending) "
                "with 114 canonical features predicting 72-hour continuous AQI. "
                "Evaluation Provenance: Out-of-time test set (9,311 samples, 2025-06-07 to 2026-08-28 UTC): "
                "overall_rmse=75.91, overall_mae=53.55, overall_r2=0.4858, h1_rmse=50.43, h72_rmse=77.43. "
                "Temporal walk-forward validation (4 folds): mean RMSE=83.44 (4/4 fold wins vs persistence, 24.46% relative gain)."
            )

        logger.info(
            f"Registering production champion bundle under name '{self.model_name}', "
            f"version {self.model_version}..."
        )
        kwargs = {
            "name": self.model_name,
            "version": self.model_version,
            "metrics": manifest["benchmarks"],
            "description": description,
            "input_example": input_df,
        }
        if model_schema is not None:
            kwargs["model_schema"] = model_schema

        model = mr.python.create_model(**kwargs)
        model.save(str(bundle_dir))
        logger.info(f"Successfully registered model '{self.model_name}' v{self.model_version} in Hopsworks Model Registry")
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
                model = mr.get_model(name=self.model_name, version=self.model_version)
                if model is None:
                    raise FeatureStoreError(
                        f"Model '{self.model_name}' version {self.model_version} not found in Hopsworks Model Registry."
                    )
                if download_dir is None:
                    download_dir = MODELS_DIR / "hopsworks_bundle"
                download_dir.mkdir(parents=True, exist_ok=True)

                downloaded_path = Path(model.download(str(download_dir)))
                manifest = self.verify_bundle_integrity(downloaded_path)
                return downloaded_path, manifest
            except ValidationError:
                raise
            except Exception as e:
                logger.warning(f"Cloud Model Registry fetch failed ({e}); activating local model fallback.")

        # Local fallback
        local_bundle_dir = MODELS_DIR / "bundle"
        has_flat_manifest = (local_bundle_dir / "manifest.json").exists()
        has_prod_manifest = (local_bundle_dir / "production" / "manifest.json").exists()
        if not has_flat_manifest and not has_prod_manifest:
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

    def verify_downloaded_parity(self, downloaded_dir: Path) -> dict[str, Any]:
        """Verify clean download integrity, independent loading, 72-horizon prediction parity, and explainability parity.

        Args:
            downloaded_dir: Directory containing downloaded model artifacts.

        Returns:
            Parity audit summary report dictionary.
        """
        manifest = self.verify_bundle_integrity(downloaded_dir)

        file_hashes = {}
        for fname, expected_hash in manifest.get("files", {}).items():
            fpath = downloaded_dir / "production" / fname
            if not fpath.exists():
                fpath = downloaded_dir / fname
            actual_hash = compute_file_sha256(fpath)
            file_hashes[fname] = actual_hash
            if actual_hash != expected_hash:
                raise ValidationError(f"Downloaded file {fname} hash mismatch: {actual_hash} != {expected_hash}")

        from src.inference.runtime_resolver import RuntimeAssetResolver
        from src.inference.model_loader import ModelLoader
        from src.inference.predictor import AQIPredictor
        from src.inference.explainer import ModelExplainer

        downloaded_resolver = RuntimeAssetResolver(mode="runtime", base_runtime_dir=downloaded_dir)
        downloaded_loader = ModelLoader(resolver=downloaded_resolver)
        downloaded_model = downloaded_loader.load_model()
        downloaded_scaler = downloaded_loader.load_scaler()
        downloaded_schema = downloaded_loader.load_schema()

        if len(downloaded_schema) != 114:
            raise ValidationError(f"Expected 114 schema features, got {len(downloaded_schema)}")

        local_resolver = RuntimeAssetResolver(mode="runtime", base_runtime_dir=DATA_DIR / "runtime")
        local_predictor = AQIPredictor(runtime_resolver=local_resolver)
        downloaded_predictor = AQIPredictor(runtime_resolver=downloaded_resolver)

        raw_features, meta = local_resolver.load_bootstrap_feature_vector()
        current_aqi = meta.get("current_aqi", 119.7)

        p_local = local_predictor.predict_72h(
            features=raw_features,
            current_aqi=current_aqi,
            input_observed_at=meta.get("input_observed_at"),
        )
        p_down = downloaded_predictor.predict_72h(
            features=raw_features,
            current_aqi=current_aqi,
            input_observed_at=meta.get("input_observed_at"),
        )

        arr_local = np.array([x["aqi"] for x in p_local["forecasts"]])
        arr_down = np.array([x["aqi"] for x in p_down["forecasts"]])

        max_diff = float(np.max(np.abs(arr_down - arr_local)))
        if max_diff > 1e-6:
            raise ValidationError(f"Prediction parity failure: max absolute difference is {max_diff}")

        boundary_horizons = [1, 6, 7, 24, 37, 38, 39, 72]
        boundary_parity = {}
        for h in boundary_horizons:
            v_loc = arr_local[h - 1]
            v_down = arr_down[h - 1]
            boundary_parity[f"h{h}"] = {
                "local_aqi": round(float(v_loc), 2),
                "downloaded_aqi": round(float(v_down), 2),
                "diff": float(abs(v_down - v_loc)),
            }

        expl_parity = {}
        expl_dir = downloaded_dir / "explainability"
        if expl_dir.exists() and (expl_dir / "explainer_manifest.json").exists():
            downloaded_explainer = ModelExplainer(
                model=downloaded_model,
                scaler=downloaded_scaler,
                feature_names=downloaded_schema,
                resolver=downloaded_resolver,
            )
            local_loader = ModelLoader(resolver=local_resolver)
            local_explainer = ModelExplainer(
                model=local_loader.load_model(),
                scaler=local_loader.load_scaler(),
                feature_names=local_loader.load_schema(),
                resolver=local_resolver,
            )

            for h in [1, 24, 72]:
                e_down = downloaded_explainer.explain_horizon(
                    raw_features.values, horizon=h, top_k=5, current_aqi=current_aqi
                )
                e_loc = local_explainer.explain_horizon(
                    raw_features.values, horizon=h, top_k=5, current_aqi=current_aqi
                )
                p_diff = abs(e_down["predicted_aqi"] - e_loc["predicted_aqi"])
                b_diff = abs(e_down["base_value"] - e_loc["base_value"])
                if p_diff > 1e-6 or b_diff > 1e-6:
                    raise ValidationError(f"Explainability parity failure at horizon {h}")
                expl_parity[f"h{h}"] = {
                    "predicted_aqi": round(float(e_down["predicted_aqi"]), 4),
                    "predicted_aqi_display": round(float(e_down["predicted_aqi"]), 1),
                    "base_value": round(float(e_down["base_value"]), 4),
                    "top_feature": e_down["top_features"][0]["feature"] if e_down.get("top_features") else None,
                    "status": "PASS",
                }

        return {
            "status": "PASS",
            "model_id": manifest.get("model_id", "EXP-019"),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_count": len(downloaded_schema),
            "output_horizons": len(arr_down),
            "file_hashes": file_hashes,
            "max_abs_prediction_diff": max_diff,
            "boundary_horizon_parity": boundary_parity,
            "explainability_parity": expl_parity,
        }


def main():
    """Command-line interface for Hopsworks Model Registry operations."""
    parser = argparse.ArgumentParser(description="Hopsworks Model Registry Production Champion Manager")
    parser.add_argument("--dry-run", action="store_true", help="Validate local bundle and inspect remote registry without changes")
    parser.add_argument("--register", action="store_true", help="Register frozen EXP-019 production champion")
    parser.add_argument("--verify", action="store_true", help="Inspect registered model details in Hopsworks Model Registry")
    parser.add_argument("--download-test", action="store_true", help="Download registered model to clean temp dir and verify parity")
    parser.add_argument("--idempotency-check", action="store_true", help="Test safe re-run idempotency without duplicate creation")
    parser.add_argument("--all", action="store_true", help="Run register, verify, download-test, and idempotency-check sequentially")

    args = parser.parse_args()
    if not any([args.dry_run, args.register, args.verify, args.download_test, args.idempotency_check, args.all]):
        parser.print_help()
        return

    connector = HopsworksModelRegistryConnector()

    if args.dry_run:
        print("\n=== Pre-Flight Dry Run Inspection ===")
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = connector.build_local_bundle(Path(td))
            manifest = connector.verify_bundle_integrity(bundle_dir)
            print(f"Local bundle validated: {len(manifest['files'])} model files verified.")
            for fname, h in manifest["files"].items():
                print(f"  - {fname}: {h}")
            print(f"Model ID: {manifest.get('model_id')}")
            print(f"Benchmarks: {manifest.get('benchmarks')}")

        if connector.is_cloud_configured():
            print("\nChecking live Model Registry...")
            mr = connector._login()
            models = mr.get_models(name=connector.model_name)
            print(f"Existing models with name '{connector.model_name}': {len(models)}")
            for m in models:
                print(f"  - Version: {m.version}, Created: {m.created}")
        else:
            print("Cloud credentials not configured; dry run completed locally.")
        return

    if args.register or args.all:
        print(f"\n=== Registering Production Champion '{connector.model_name}' v{connector.model_version} ===")
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = connector.build_local_bundle(Path(td))
            model = connector.register_production_champion(bundle_dir)
            print(f"Model successfully registered: {model.name} v{model.version}")

    if args.verify or args.all:
        print(f"\n=== Verifying Model Registry for '{connector.model_name}' ===")
        mr = connector._login()
        model = mr.get_model(name=connector.model_name, version=connector.model_version)
        if model is None:
            raise RuntimeError(f"Model '{connector.model_name}' v{connector.model_version} not found in registry!")
        print(f"Model Name: {model.name}")
        print(f"Version: {model.version}")
        print(f"ID: {model.id}")
        print(f"Description: {model.description}")
        print(f"Metrics: {getattr(model, 'training_metrics', getattr(model, 'metrics', None))}")
        print(f"Framework: {getattr(model, 'framework', 'generic')}")
        print(f"Created: {model.created}")

    if args.download_test or args.all:
        print("\n=== Clean Download & Prediction Parity Test ===")
        with tempfile.TemporaryDirectory() as td:
            download_dir = Path(td) / "downloaded_model"
            mr = connector._login()
            model = mr.get_model(name=connector.model_name, version=connector.model_version)
            model.download(str(download_dir))
            parity_report = connector.verify_downloaded_parity(download_dir)
            print(f"Clean download parity status: {parity_report['status']}")
            print(f"Max absolute prediction difference: {parity_report['max_abs_prediction_diff']:.10f}")
            print("Boundary horizons comparison:")
            for h, vals in parity_report["boundary_horizon_parity"].items():
                print(f"  {h}: local={vals['local_aqi']} vs downloaded={vals['downloaded_aqi']} (diff={vals['diff']})")
            if parity_report.get("explainability_parity"):
                print("Explainability parity:")
                for h, vals in parity_report["explainability_parity"].items():
                    print(f"  {h}: predicted={vals['predicted_aqi']}, base={vals['base_value']}, top={vals['top_feature']} [{vals['status']}]")

    if args.idempotency_check or args.all:
        print("\n=== Safe Re-Run / Idempotency Test ===")
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = connector.build_local_bundle(Path(td))
            model_rerun = connector.register_production_champion(bundle_dir)
            assert model_rerun.version == connector.model_version, f"Version incremented to {model_rerun.version}!"
            print(f"Re-run verified: existing version {model_rerun.version} reused safely without duplicate versioning.")

    print("\nAll requested Model Registry operations completed successfully!")


if __name__ == "__main__":
    main()
