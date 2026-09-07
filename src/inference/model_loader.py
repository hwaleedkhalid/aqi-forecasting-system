"""Pearls AQI Predictor - Model Loader for Production Artifacts.

Thread-safe loading and memory-caching of the production hybrid model,
feature scaler, and canonical feature schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.exceptions import ValidationError
from src.inference.runtime_resolver import RuntimeAssetResolver
from src.logger import logger
from src.models.hybrid_specialist_model import PersistenceAwareHybridModel


class ModelLoader:
    """Loads and validates frozen production model artifacts."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        scaler_path: Path | str | None = None,
        schema_path: Path | str | None = None,
        resolver: RuntimeAssetResolver | None = None,
    ) -> None:
        """Initialize ModelLoader with artifact paths.

        Args:
            model_path: Path to serialized production model (default from RuntimeAssetResolver).
            scaler_path: Path to fitted StandardScaler (default from RuntimeAssetResolver).
            schema_path: Path to feature schema JSON (default from RuntimeAssetResolver).
            resolver: Optional RuntimeAssetResolver instance.
        """
        self.resolver = resolver or RuntimeAssetResolver()
        self.model_path = Path(model_path or self.resolver.get_model_path())
        self.scaler_path = Path(scaler_path or self.resolver.get_scaler_path())
        self.schema_path = Path(schema_path or self.resolver.get_schema_path())

        self._model: PersistenceAwareHybridModel | None = None
        self._scaler: StandardScaler | None = None
        self._feature_schema: list[str] | None = None

    def load_schema(self) -> list[str]:
        """Load and cache canonical ordered feature column names.

        Returns:
            List of 114 feature column names in canonical order.
        """
        if self._feature_schema is not None:
            return self._feature_schema

        if not self.schema_path.exists():
            raise FileNotFoundError(f"Feature schema file not found at {self.schema_path}")

        with open(self.schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        feature_names = data.get("feature_names", [])
        if not feature_names:
            raise ValidationError("Feature schema is empty or missing 'feature_names' key.")

        self._feature_schema = list(feature_names)
        logger.info(f"Loaded feature schema with {len(self._feature_schema)} features.")
        return self._feature_schema

    def load_scaler(self) -> StandardScaler:
        """Load and cache fitted StandardScaler.

        Returns:
            Fitted StandardScaler instance expecting 114 input features.
        """
        if self._scaler is not None:
            return self._scaler

        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Feature scaler artifact not found at {self.scaler_path}")

        scaler = joblib.load(self.scaler_path)
        if not isinstance(scaler, StandardScaler):
            raise ValidationError(f"Expected StandardScaler artifact, got {type(scaler).__name__}")

        schema = self.load_schema()
        if scaler.n_features_in_ != len(schema):
            raise ValidationError(
                f"Scaler feature count mismatch: scaler expects {scaler.n_features_in_}, "
                f"schema has {len(schema)}."
            )

        self._scaler = scaler
        logger.info(f"Loaded feature scaler expecting {scaler.n_features_in_} features.")
        return self._scaler

    def load_model(self) -> PersistenceAwareHybridModel:
        """Load and cache frozen production model artifact.

        The artifact is a serialized component dictionary containing:
          - m_short: LightGBMDirectMultiOutput (h1..6)
          - m_rest: Ridge (h7..72)
          - min_blend_weight: float
          - current_aqi_col_idx: int (index 9 in schema)

        Returns:
            Reconstructed PersistenceAwareHybridModel instance.
        """
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise FileNotFoundError(f"Production model artifact not found at {self.model_path}")

        model = PersistenceAwareHybridModel().load(self.model_path)
        if not model.is_fitted:
            raise RuntimeError("Loaded model failed is_fitted validation.")

        schema = self.load_schema()
        # Verify Ridge component expects the full 114-column vector
        if hasattr(model.m_rest, "n_features_in_"):
            if model.m_rest.n_features_in_ != len(schema):
                raise ValidationError(
                    f"Model input feature count mismatch: model expects {model.m_rest.n_features_in_}, "
                    f"schema has {len(schema)}."
                )

        self._model = model
        logger.info("Loaded production hybrid model (EXP-019) successfully.")
        return self._model

    def validate_feature_vector(self, df: Any) -> None:
        """Validate that input feature columns match the canonical schema in name and order.

        Args:
            df: DataFrame containing input telemetry features.

        Raises:
            ValidationError: If columns are missing, extra, or incorrectly ordered.
        """
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            return

        expected = self.load_schema()
        actual = [c for c in expected if c in df.columns]

        missing = set(expected) - set(df.columns)
        if missing:
            raise ValidationError(f"Input features missing {len(missing)} expected columns: {sorted(list(missing))[:5]}")

        # Check order of expected columns
        df_order = [c for c in df.columns if c in expected]
        if df_order != expected:
            raise ValidationError("Input feature column order does not match canonical schema.")
