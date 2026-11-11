"""
SHAP Tree Explainer & Telemetry Feature Contributions.

Computes exact Shapley feature contributions using either TreeExplainer (if shap is installed)
or XGBoost's native C++ Tree SHAP kernel (booster.predict(pred_contribs=True)).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import xgboost as xgb


class TurbofanExplainer:
    """
    Computes local and global Shapley attribution values for predictive models.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.feature_cols = classifier.feature_cols

    def compute_shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute Shapley feature contributions matrix of shape (N, num_features).
        Uses native booster prediction for maximum speed and zero external dependencies.
        """
        X = df[self.feature_cols].fillna(0.0).to_numpy()
        booster = self.classifier.model.get_booster()
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_cols)

        # pred_contribs=True returns (N, num_features + 1), where last column is bias
        contribs = booster.predict(dmatrix, pred_contribs=True)
        shap_values = contribs[:, :-1]
        return shap_values

    def get_top_contributors(
        self,
        row_df: pd.DataFrame,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Extract top-k most influential sensor features pushing probability toward failure.
        """
        shap_vals = self.compute_shap_values(row_df)[0]
        indexed = list(zip(self.feature_cols, shap_vals))
        # Sort by positive contribution to failure
        sorted_factors = sorted(indexed, key=lambda x: x[1], reverse=True)

        results = []
        for feat, val in sorted_factors[:top_k]:
            results.append({
                "feature": feat,
                "shap_value": float(val),
                "raw_value": float(row_df[feat].iloc[0]) if feat in row_df.columns else None,
            })
        return results

