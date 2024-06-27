"""AutoML class for exploring a proper setup for the AutoML class.

takes the shape of the example AutoML script that simply returns dummy predictions.
"""
from __future__ import annotations

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging

from automl.algorithm_search import return_model as Model

logger = logging.getLogger(__name__)

METRICS = {"r2": r2_score}

class AutoML:

    def __init__(
        self,
        seed: int,
        metric: str = "r2",
    ) -> None:
        self.seed = seed
        self.metric = METRICS[metric]
        self._model: Model | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> AutoML:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            random_state=self.seed,
            test_size=0.2,
        )


        ### imoplement alg search here, now just return a model
        model = Model()
        model.fit(X_train, y_train)
        self._model = model

        val_preds = model.predict(X_val)
        val_score = self.metric(y_val, val_preds)
        logger.info(f"Validation score: {val_score:.4f}")

        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted")

        return self._model.predict(X)  # type: ignore
