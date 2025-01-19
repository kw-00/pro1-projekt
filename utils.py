from typing import Any, Dict, List, Tuple, TypedDict
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, GridSearchCV

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


def remove_and_fill_non_numerical(df: pd.DataFrame, values_counting_as_na: List[str]) -> None:
    for v in values_counting_as_na:
        df.replace(v, np.nan, inplace=True)

    for column_name in df.dropna().select_dtypes(include=["number"]):
        df[column_name] = df[column_name].fillna(df.dropna()[column_name].mean())

    for column_name in df.dropna().select_dtypes(exclude=["number"]):
        df.drop(labels=[column_name], axis="columns", inplace=True)

    
        
