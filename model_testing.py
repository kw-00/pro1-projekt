# Imports and data
from typing import Dict
import pandas as pd
import numpy as np

# Training

# Regressors
from sklearn.linear_model import ElasticNetCV, LinearRegression, LassoCV, RidgeCV 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Data manipulation
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA


# Metrics
from sklearn.metrics import mean_squared_error

# Other
import pickle

# Loading and processing data
def test(data: pd.DataFrame, explained_variance: float | None = None, n_components: int | None = None) -> Dict[str, GridSearchCV]:
    if explained_variance is not None and n_components is not None:
        raise ValueError(
            "explained_variance and n_components cannot both have values assigned — one of them has to be None."
        )

    # Train/test split
    X = data.drop(data.columns[-1], axis="columns")
    y = data[data.columns[-1]]
    cv = KFold()

    # Compress dataset
    if explained_variance is not None:
        compressor = data_processing_utils.PCACompressor(X, explained_variance=explained_variance)
        X = compressor.transform(X)
    elif n_components is not None:
        compressor = data_processing_utils.PCACompressor(X, n_components=n_components)
        X = compressor.transform(X)


    # Preparing candidate estimators and parameter grids

    # Regressors that don't use search
    linear_regression = LinearRegression()
    lasso = LassoCV()
    ridge = RidgeCV()
    elastic_net = ElasticNetCV()

    # Regressors using search
    decision_tree_regressor = DecisionTreeRegressor()
    random_forest_regressor = RandomForestRegressor()
    extra_trees_regressor = ExtraTreesRegressor()
    gradient_boosting_regressor = GradientBoostingRegressor()
    mlp_regressor = MLPRegressor()


    scoring = "neg_mean_squared_error"

    feature_count = X.shape[1]

    decision_tree_regressor_grid = {
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 0.01, 0.1],
        "min_samples_leaf": [1, 0.01, 0.1],
        "max_features": [1, 2, 5, 0.5, feature_count]
    }

    random_forest_regressor_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 0.01, 0.1],
        "min_samples_leaf": [1, 0.01, 0.1],
        "max_features": [1, 2, 5, 0.5, feature_count],
        "n_estimators": [100, 200, 500],
    }

    extra_trees_regressor_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 0.01, 0.1],
        "min_samples_leaf": [1, 0.01, 0.1],
        "max_features": [1, 2, 5, 0.5, feature_count],
        "n_estimators": [100, 200, 500],
    }

    gradient_boosting_regressor_grid = {
        "n_estimators": [100, 200, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1]
    }

    mlp_regressor_grid = {
        "activation": ["relu", "identity", "tanh"],
        "hidden_layer_sizes": [
                (64, 32, 16), (128, 64, 32), (256, 128, 64),
                (128, 64, 32, 16),
                (256, 128, 64, 32), (256, 128, 64, 32, 16)
            ],
        "max_iter": [1000]
    }

    linear_regression_search = GridSearchCV(estimator=linear_regression, param_grid={}, cv=cv, scoring=scoring, n_jobs=-1)
    lasso_search = GridSearchCV(estimator=lasso, param_grid={"max_iter": [2000]}, cv=cv, scoring=scoring, n_jobs=-1)
    ridge_search = GridSearchCV(estimator=ridge, param_grid={}, cv=cv, scoring=scoring, n_jobs=-1)
    elastic_net_search = GridSearchCV(estimator=elastic_net, param_grid={"max_iter": [2000]}, cv=cv, scoring=scoring, n_jobs=-1)
    decision_tree_regressor_search = GridSearchCV(estimator=decision_tree_regressor, param_grid=decision_tree_regressor_grid, cv=cv, scoring=scoring, n_jobs=-1)
    random_forest_regressor_search = GridSearchCV(estimator=random_forest_regressor, param_grid=random_forest_regressor_grid, cv=cv, scoring=scoring, n_jobs=-1)
    extra_trees_regressor_search = GridSearchCV(estimator=extra_trees_regressor, param_grid=extra_trees_regressor_grid, cv=cv, scoring=scoring, n_jobs=-1)
    gradient_boosting_regressor_search = GridSearchCV(estimator=gradient_boosting_regressor, param_grid=gradient_boosting_regressor_grid, cv=cv, scoring=scoring, n_jobs=-1)
    mlp_regressor_search = GridSearchCV(estimator=mlp_regressor, param_grid=mlp_regressor_grid, cv=cv, scoring=scoring, n_jobs=-1)

    search_results: dict[str, GridSearchCV] = {
        "linear_regression_search": linear_regression_search,
        "lasso_search": lasso_search,
        "ridge_search": ridge_search,
        "elastic_net": elastic_net_search,
        # "decision_tree_regressor_search": decision_tree_regressor_search,
        # "random_forest_regressor_search": random_forest_regressor_search,
        # "extra_trees_regressor_search": extra_trees_regressor_search,
        # "gradient_boosting_regressor_search": gradient_boosting_regressor_search,  
        # "mlp_regressor_search": mlp_regressor_search
    }

    variance = y.var()
    for k, v in search_results.items():
        v.fit(X=X, y=y)
        print(f"{k} best score: ".ljust(50), -v.best_score_ / variance) # type: ignore
    
    return search_results


# Import data
import pandas as pd
import model_testing
import data_processing_utils

data = pd.read_csv("communities.data", header=None)

# Drop categorical columns
data.drop([0, 1, 2, 3], axis="columns", inplace=True)

# Replace na values and drop non-numerical columns
data_processing_utils.remove_and_fill_non_numerical(data)

tests = {}
for explained_variance in [0.8, 0.9, 0.95, 0.98, 0.9999, None]:
    print(f"explained_variance={explained_variance}")
    search_results = model_testing.test(data, explained_variance=explained_variance)
    tests[explained_variance] = search_results
    print("\n")