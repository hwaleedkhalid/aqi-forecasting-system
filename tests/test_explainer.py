"""Unit, mathematical additivity, boundary routing, and API tests for SHAP model explainability."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.api.app import create_app
from src.dashboard.components import build_feature_attribution_figure, render_explainability_section
from src.exceptions import ValidationError
from src.inference.explainer import EXPLAINABILITY_DIR, ModelExplainer
from src.inference.model_loader import ModelLoader
from src.inference.predictor import AQIPredictor


@pytest.fixture
def predictor() -> AQIPredictor:
    """Create singleton AQIPredictor instance."""
    return AQIPredictor()


@pytest.fixture
def explainer(predictor) -> ModelExplainer:
    """Create ModelExplainer instance with loaded EXP-019 model and schema."""
    model = predictor.model_loader.load_model()
    scaler = predictor.model_loader.load_scaler()
    schema = predictor.model_loader.load_schema()
    return ModelExplainer(model=model, scaler=scaler, feature_names=schema)


@pytest.fixture
def sample_feature_vector(predictor) -> np.ndarray:
    """Get sample 114-feature row from dataset."""
    import json
    import pandas as pd
    from src.config import PROCESSED_DATA_DIR, RUNTIME_DIR
    schema = predictor.model_loader.load_schema()
    csv_file = PROCESSED_DATA_DIR / "features_v2_weather.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file).tail(1)
        return df[schema].values
    bootstrap_file = RUNTIME_DIR / "bootstrap" / "latest_feature_vector.json"
    if bootstrap_file.exists():
        with open(bootstrap_file, "r", encoding="utf-8") as f:
            bdata = json.load(f)
        bfeats = bdata.get("features", {})
        return np.array([[bfeats.get(k, 0.0) for k in schema]], dtype=np.float64)
    np.random.seed(42)
    return np.random.uniform(10.0, 50.0, (1, len(schema)))


@pytest.fixture
def flask_client():
    """Create Flask test client."""
    app = create_app({"TESTING": True})
    return app.test_client()


class TestSpecialistBoundaryRouting:
    """Test suite ensuring exact routing matches the frozen EXP-019 hybrid architecture."""

    def test_horizon_1_and_6_route_to_lightgbm(self, explainer, sample_feature_vector):
        exp1 = explainer.explain_horizon(sample_feature_vector, horizon=1, current_aqi=150.0)
        assert exp1["specialist_type"] == "LightGBM Tree Specialist"
        assert exp1["blend_weight"] == 1.0
        assert exp1["persistence_component"] == 0.0

        exp6 = explainer.explain_horizon(sample_feature_vector, horizon=6, current_aqi=150.0)
        assert exp6["specialist_type"] == "LightGBM Tree Specialist"
        assert exp6["blend_weight"] == 1.0
        assert exp6["persistence_component"] == 0.0

    def test_horizon_7_and_37_route_to_pure_ridge(self, explainer, sample_feature_vector):
        exp7 = explainer.explain_horizon(sample_feature_vector, horizon=7, current_aqi=150.0)
        assert exp7["specialist_type"] == "Ridge Linear Specialist"
        assert exp7["blend_weight"] == 1.0
        assert exp7["persistence_component"] == 0.0

        exp37 = explainer.explain_horizon(sample_feature_vector, horizon=37, current_aqi=150.0)
        assert exp37["specialist_type"] == "Ridge Linear Specialist"
        assert exp37["blend_weight"] == 1.0
        assert exp37["persistence_component"] == 0.0

    def test_horizon_38_blend_boundary_has_weight_one_and_zero_persistence(self, explainer, sample_feature_vector):
        exp38 = explainer.explain_horizon(sample_feature_vector, horizon=38, current_aqi=150.0)
        assert exp38["specialist_type"] == "Ridge + Persistence Blend"
        assert exp38["blend_weight"] == 1.0
        assert exp38["persistence_component"] == 0.0

    def test_horizon_39_blend_has_decayed_weight_and_positive_persistence(self, explainer, sample_feature_vector):
        exp39 = explainer.explain_horizon(sample_feature_vector, horizon=39, current_aqi=150.0)
        assert exp39["specialist_type"] == "Ridge + Persistence Blend"
        assert exp39["blend_weight"] < 1.0
        assert exp39["persistence_component"] > 0.0

    def test_horizon_72_blend_has_minimum_weight_and_thirty_percent_persistence(self, explainer, sample_feature_vector):
        exp72 = explainer.explain_horizon(sample_feature_vector, horizon=72, current_aqi=100.0)
        assert exp72["specialist_type"] == "Ridge + Persistence Blend"
        assert exp72["blend_weight"] == 0.7
        assert exp72["persistence_component"] == pytest.approx(30.0, abs=1e-3)


class TestMathematicalAdditivityAndDecomposition:
    """Test suite ensuring exact reconstruction of preclip output across all horizons."""

    @pytest.mark.parametrize("h", [1, 6, 7, 24, 37, 38, 39, 48, 72])
    def test_exact_preclip_additivity(self, explainer, sample_feature_vector, h):
        current_aqi = 125.0
        exp = explainer.explain_horizon(sample_feature_vector, horizon=h, current_aqi=current_aqi)

        w = exp["blend_weight"]
        spec_base = exp["specialist_base_value"]
        base = exp["base_value"]
        raw_spec = exp["raw_specialist_output"]
        model_comp = exp["model_component"]
        pers_comp = exp["persistence_component"]
        preclip = exp["explained_output_preclip"]
        pred_aqi = exp["predicted_aqi"]
        err = exp["additivity_error"]

        # 1. Base value scaling (allowing small display-rounding margin)
        assert base == pytest.approx(w * spec_base, abs=5e-2)

        # 2. Model component scaling
        assert model_comp == pytest.approx(w * raw_spec, abs=5e-2)

        # 3. Persistence component definition
        assert pers_comp == pytest.approx((1.0 - w) * current_aqi, abs=5e-2)

        # 4. Composite preclip reconstruction
        assert preclip == pytest.approx(model_comp + pers_comp, abs=5e-2)

        # 5. Non-negative output bounding
        assert pred_aqi == pytest.approx(max(0.0, preclip), abs=1e-4)

        # 6. Strict additivity error < 1e-4
        assert err < 1e-4

    def test_feature_attribution_structure_and_neutral_impact_labels(self, explainer, sample_feature_vector):
        exp = explainer.explain_horizon(sample_feature_vector, horizon=24, top_k=10)
        top_feats = exp["top_features"]
        assert len(top_feats) == 10

        for feat in top_feats:
            assert "feature" in feat
            assert "raw_value" in feat
            assert "scaled_value" in feat
            assert "shap_value" in feat
            assert feat["impact"] in ["increases_prediction", "decreases_prediction"]
            if feat["shap_value"] >= 0:
                assert feat["impact"] == "increases_prediction"
            else:
                assert feat["impact"] == "decreases_prediction"


class TestAPIAndPredictorConsistency:
    """Test suite ensuring exact consistency between explainer, predictor, and Flask REST API."""

    def test_explainer_and_predictor_produce_identical_forecast_aqi(self, predictor):
        """Assert /api/explain.predicted_aqi == /api/forecast.forecasts[h-1].aqi for long horizon h72."""
        forecast_res = predictor.predict_latest(use_cache=False)
        forecast_h72 = forecast_res["forecasts"][71]["aqi"]

        explain_res = predictor.explain_latest(horizon=72)
        explain_h72 = explain_res["predicted_aqi"]

        assert explain_h72 == pytest.approx(forecast_h72, abs=0.1)
        assert round(explain_h72, 1) == forecast_h72
        assert explain_res["input_observed_at"] == forecast_res["input_observed_at"]
        assert explain_res["forecast_origin"] == forecast_res["forecast_origin"]
        assert explain_res["is_stale"] == forecast_res["is_stale"]

    def test_api_explain_endpoint_200_ok(self, flask_client):
        resp = flask_client.get("/api/explain?horizon=24&top_k=5")
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["model_id"] == "EXP-019"
        assert data["horizon"] == 24
        assert len(data["top_features"]) == 5
        assert "global_persistence_mean_contribution" in data
        assert "global_top_features" in data
        assert "additivity_error" in data

    def test_api_explain_endpoint_invalid_params_400(self, flask_client):
        # Out of range horizon
        resp = flask_client.get("/api/explain?horizon=0")
        assert resp.status_code == 400
        assert "Invalid 'horizon'" in resp.get_json()["error"]

        resp = flask_client.get("/api/explain?horizon=73")
        assert resp.status_code == 400

        # Invalid top_k
        resp = flask_client.get("/api/explain?horizon=24&top_k=0")
        assert resp.status_code == 400

        resp = flask_client.get("/api/explain?horizon=24&top_k=100")
        assert resp.status_code == 400


class TestExplainabilityUIComponents:
    """Test suite for Streamlit UI explainability chart builders and layout functions."""

    def test_build_feature_attribution_figure_structure(self, explainer, sample_feature_vector):
        exp = explainer.explain_horizon(sample_feature_vector, horizon=24, top_k=10)
        fig = build_feature_attribution_figure(exp["top_features"], horizon=24)

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"
        assert fig.data[0].orientation == "h"
        assert len(fig.data[0].x) == 10

    def test_render_explainability_section_callable(self, explainer, sample_feature_vector):
        exp = explainer.explain_horizon(sample_feature_vector, horizon=24, top_k=10)
        exp["global_persistence_mean_contribution"] = 17.4
        exp["global_top_features"] = [{"feature": "pm10", "mean_abs_shap": 85.7}]

        with (
            patch("streamlit.subheader"),
            patch("streamlit.caption"),
            patch("streamlit.columns", return_value=[MagicMock()]*5),
            patch("streamlit.plotly_chart"),
            patch("streamlit.expander"),
            patch("streamlit.markdown"),
            patch("streamlit.dataframe"),
        ):
            render_explainability_section(exp)
