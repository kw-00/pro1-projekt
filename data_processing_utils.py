from tkinter import N
from typing import Any, Dict, List, Tuple, TypedDict
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, GridSearchCV
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


def select_estimator(
        estimators_and_param_grids: List[Tuple[List[BaseEstimator], Dict[str, Any]]],
        cv: BaseCrossValidator | BaseShuffleSplit,
        X: Any,
        y: Any,
        scoring: str | None = None,
    ) \
        -> List[GridSearchCV]:
    
    grid_searches: List[GridSearchCV] = []
    for estimators, param_grid in estimators_and_param_grids:
        for estimator in estimators:
            gscv = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv
            )
            grid_searches.append(gscv)

    for gs in grid_searches:
        gs.fit(X=X, y=y)

    return grid_searches


def remove_and_fill_non_numerical(df: pd.DataFrame) -> None:
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].mean())
    
    df.dropna(axis="columns", inplace=True)

class PCACompressor:
    def __init__(self, X: np.ndarray | pd.DataFrame, explained_variance: float | None = None, n_components: int | None = None) -> None:
        if explained_variance is None and n_components is None:
            raise ValueError("explained_variance and n_components cannot both be None.")
        
        if explained_variance is not None and n_components is not None:
            raise ValueError(
                "explained_variance and n_components cannot both have values assigned — one of them has to be None."
            )

        if explained_variance is not None:
            pca = PCA()
            pca.fit(X)
            cumulative_sum = np.cumsum(pca.explained_variance_ratio_)

            cumsum_ge_variance = cumulative_sum >= explained_variance
            if True not in cumsum_ge_variance:
                n = X.shape[1]
            else:
                n = np.argmax(cumsum_ge_variance) + 1

            self._pca = PCA(n_components=n)
            self._pca.fit(X)
        else:
            self._pca = PCA(n_components=n_components)
            self._pca.fit(X)

    def transform(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self._pca.transform(X))

    
        
