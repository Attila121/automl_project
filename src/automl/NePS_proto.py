import logging
from torch import nn
import pandas as pd
import numpy as np

import neps

from __future__ import annotations

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

METRICS = {"r2": r2_score}


class NePS_AutoML:
    def __init__(
        self,
        seed: int,
        metric: str = "r2",
        pipeline_space: dict = {
        "learning_rate": neps.FloatParameter(lower=0.00001, upper=0.1, log=True),
        "num_epochs": neps.IntegerParameter(lower=1, upper=5, is_fidelity=True),
        "optimizer": neps.CategoricalParameter(choices=["adam", "sgd", "rmsprop"]),
        "dropout_rate": neps.FloatParameter(value=0.5),
        },
        model: Module = None,
    ) -> None:
        self.seed = seed
        self.metric = METRICS[metric]
        self._model: Module | None = None
        self.pipeline_space = pipeline_space

    def fit(
            self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> NePS_AutoML:
        
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            random_state=self.seed,
            test_size=0.2,
        )

        neps.run(
            pipeline_space=pipeline_space,
            run_pipeline=run_pipeline,
            num_iterations=10,
            output_path="results_NePS.csv",
            minimize=True,
            # searcher =..., we let NePS choose the searcher
        )
    
    


    def run_pipeline(
            lr: float,
            num_epochs: int,
            optimizer: str,
            dropout_rate: float,
    ) -> dict:
        start = time.time()

        # insert here your own model
        model = MyModel(architecture_parameter)

        # insert here your training/evaluation pipeline
        validation_error, training_error = train_and_eval(
        model, lr, num_epochs, optimizer, dropout_rate
    )

        end = time.time()
        duration = end - start

        logger.info(f"duration: {duration:.4f}")

        return loss
    
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # return the prediction of the most promising model and its configuration
        if self._model is None:
            raise ValueError("Model not fitted")
        
        return self._model.predict(X)
