
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import joblib
import numpy as np

def run_hyperparameter_search(param_grid, X_depth_train, y_depth_train, weights_depth, test, train, features, year, lp):

    param_list = list(ParameterGrid(param_grid))
    results = []

    for params in param_list:
        suffix = "_".join(f"{k}-{v}" for k, v in params.items())
        print(suffix)

        rf_depth = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        rf_depth.fit(X_depth_train, y_depth_train, sample_weight=weights_depth)

        X_test = test[features]
        y_depth_test = test['depth_water']
        y_depth_pred = rf_depth.predict(X_test)
        y_depth_pred_train = rf_depth.predict(X_depth_train)

        current_date = datetime.now().strftime("%Y%m%d")
        model_path = f"output/rf_depth_model_hold_out_{year}_{current_date}_{suffix}.joblib"
        joblib.dump(rf_depth, model_path)

        mse_depth = mean_squared_error(y_depth_test, y_depth_pred)
        print(f'Mean Squared Error for depth prediction: {mse_depth}')

        test_results = test.copy()
        test_results['depth_water_pred'] = y_depth_pred

        train_results = train.copy()
        train_results['depth_water_pred'] = np.nan
        train_results.loc[X_depth_train.index, 'depth_water_pred'] = y_depth_pred_train

        lp.plot_evaluation_scatter(train_results, test_results, year, lp, 'figures/RF/hyperparameter_search/', suffix=suffix)

        score = rf_depth.score(X_test, y_depth_test)
        results.append((params, score))

    results.sort(key=lambda x: -x[1])
    return results[0][0]  # best_params
