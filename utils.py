from typing import Any, Dict, List, Tuple, TypedDict
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, GridSearchCV
from sklearn.utils._typedefs import MatrixLike



def select_estimator(
        estimators_and_param_grids: List[Tuple[List[BaseEstimator], Dict[str, Any]]],
        cv: BaseCrossValidator | BaseShuffleSplit,
        X: MatrixLike,
        y: MatrixLike,
        scoring: str | None = None,
    ) \
        -> BaseEstimator:
    
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

    best_search = max(grid_searches, key=lambda gs: gs.best_score_)
    return best_search.best_estimator_

    
        
