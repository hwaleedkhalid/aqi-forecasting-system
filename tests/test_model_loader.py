"""Unit tests for src.inference.model_loader."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from sklearn.preprocessing import StandardScaler

from src.exceptions import ValidationError
from src.inference.model_loader import ModelLoader
from src.models.hybrid_specialist_model import PersistenceAwareHybridModel


class TestModelLoader:
    """Test suite for ModelLoader."""

    def test_load_schema_returns_canonical_114_features(self):
        loader = ModelLoader()
        schema = loader.load_schema()
        assert isinstance(schema, list)
        assert len(schema) == 114
        assert "epa_aqi" in schema
        assert "pm2_5" in schema
        assert "temperature_2m" in schema
        assert "wind_speed_10m" in schema
        assert schema[0] == "co"
        assert schema[4] == "dt"
        assert schema[9] == "epa_aqi_lag_1h"

    def test_load_schema_caches_result(self):
        loader = ModelLoader()
        s1 = loader.load_schema()
        s2 = loader.load_schema()
        assert s1 is s2

    def test_load_scaler_matches_schema_dimension(self):
        loader = ModelLoader()
        scaler = loader.load_scaler()
        schema = loader.load_schema()
        assert isinstance(scaler, StandardScaler)
        assert scaler.n_features_in_ == len(schema)
        assert scaler.n_features_in_ == 114

    def test_load_model_returns_fitted_production_hybrid(self):
        loader = ModelLoader()
        model = loader.load_model()
        assert isinstance(model, PersistenceAwareHybridModel)
        assert model.is_fitted is True
        assert hasattr(model, "m_short")
        assert hasattr(model, "m_rest")
        assert model.current_aqi_col_idx == 9

    def test_missing_files_raise_appropriate_exceptions(self, tmp_path: Path):
        loader = ModelLoader(
            model_path=tmp_path / "nonexistent_model.joblib",
            scaler_path=tmp_path / "nonexistent_scaler.joblib",
            schema_path=tmp_path / "nonexistent_schema.json",
        )
        with pytest.raises(FileNotFoundError):
            loader.load_schema()
        with pytest.raises(FileNotFoundError):
            loader.load_scaler()
        with pytest.raises(FileNotFoundError):
            loader.load_model()

    def test_validate_feature_vector_passes_on_canonical_columns(self):
        import pandas as pd
        loader = ModelLoader()
        schema = loader.load_schema()
        df = pd.DataFrame([range(len(schema))], columns=schema)
        loader.validate_feature_vector(df)

    def test_validate_feature_vector_fails_on_missing_columns(self):
        import pandas as pd
        loader = ModelLoader()
        schema = loader.load_schema()
        # Drop first column
        df = pd.DataFrame([range(len(schema) - 1)], columns=schema[1:])
        with pytest.raises(ValidationError, match="missing"):
            loader.validate_feature_vector(df)

    def test_validate_feature_vector_fails_on_scrambled_column_order(self):
        """Feature ORDER matters: swapped columns must fail validation."""
        import pandas as pd
        loader = ModelLoader()
        schema = loader.load_schema()
        scrambled = list(schema)
        # Swap first two columns
        scrambled[0], scrambled[1] = scrambled[1], scrambled[0]
        df = pd.DataFrame([range(len(scrambled))], columns=scrambled)
        with pytest.raises(ValidationError, match="order does not match"):
            loader.validate_feature_vector(df)
