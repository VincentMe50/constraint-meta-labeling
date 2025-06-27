import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import multiprocessing


def primary_model(asset_idx, full_asset_data, continuous_labels_for_asset, train_ratio, params, param_grid, pre_tuned_params=None): 
    X_asset = np.array(full_asset_data) 
    y_asset_continuous = continuous_labels_for_asset # from label_mat

    n_total = len(X_asset)
    n_train = int(n_total * train_ratio) 

    X_train_pm = X_asset[:n_train] 
    X_test_pm = X_asset[n_train:] 
    y_train_pm = y_asset_continuous[:n_train] 
    y_test_pm = y_asset_continuous[n_train:]
    # y_test_pm is used by the secondary model later as the "true" primary model target for meta_label generation

    scaler_pm = StandardScaler() 
    X_train_pm_scaled = scaler_pm.fit_transform(X_train_pm) 
    X_test_pm_scaled = scaler_pm.transform(X_test_pm) 

    #dtrain_pm = xgb.DMatrix(X_train_pm_scaled, label=y_train_pm) 
    #dtest_pm = xgb.DMatrix(X_test_pm_scaled)
    #dtest_pm.set_info(device='cuda:0')
    """If there are pre_tuned parameters, skip the tuning"""
    if pre_tuned_params:
        final_params = {**params, **pre_tuned_params}
    else:
        xgb_model = xgb.XGBRegressor(random_state=42)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1, verbose=2)
        grid_search.fit(X_train_pm_scaled, y_train_pm)

        print("最优参数：", grid_search.best_params_)
        print("最优分数（负均方误差）：", grid_search.best_score_)

        best_params_primary = grid_search.best_params_
        final_params = {**params, **grid_search.best_params_}

    model_pm = xgb.XGBRegressor(**final_params, random_state=42)
    model_pm.fit(X_train_pm_scaled, y_train_pm)

    # TODO: Implement proper time-series cross-validation for num_boost_round selection if not using early stopping
    """model_pm = xgb.train(params, dtrain_pm, num_boost_round=150, # Adjusted, tune this 
                         evals=[(dtrain_pm, 'train')], verbose_eval=False) # Example for monitoring"""

    y_pred_primary_on_test = model_pm.predict(X_test_pm_scaled) 
    mse = mean_squared_error(y_test_pm, y_pred_primary_on_test)
    print("测试集均方误差：", mse)
    
    # We need y_pred_primary_on_train as well if secondary model trains on primary model's train performance
    y_pred_primary_on_train = model_pm.predict(X_train_pm_scaled) # Or on a validation set

    return y_pred_primary_on_test, scaler_pm, model_pm, (best_params_primary if not pre_tuned_params else None) # Return scaler for secondary model if features are shared