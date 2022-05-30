import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor, Pool, cv as cat_cv, sum_models
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import shap


def prepare_data(
        fut_file_path,
        etf_file_path=None,
        opt_file_path=None,
        periods=(1, 2, 3, 4, 38, 39, 40, 41)):
    # raw data
    columns = ['VOLUME_fut']
    fut_ts = pd.read_csv(fut_file_path).set_index('Time')
    ts = fut_ts
    if etf_file_path:
        etf_ts = pd.read_csv(etf_file_path).set_index('Time')
        ts = fut_ts.join(etf_ts, rsuffix='_etf', how='inner')
        columns.append('VOLUME_etf')
    if opt_file_path:
        opt_ts = pd.read_csv(opt_file_path).set_index('Time')
        ts = ts.join(opt_ts, rsuffix='_opt', how='inner')
        columns.append('VOLUME_opt')
    ts = ts.rename(columns={'VOLUME': 'VOLUME_fut'})

    # preprocessing
    test_size = 0.1
    val_size = 0.1

    val_len = int(ts.shape[0] * val_size + 0.5)
    test_len = int(ts.shape[0] * test_size + 0.5)
    train_len = int(ts.shape[0] - val_len - test_len + 0.5)

    train_indexes = list(range(train_len))
    val_indexes = list(range(train_len, train_len + val_len))
    test_indexes = list(range(train_len + val_len, train_len + val_len + test_len))

    std_num = 4
    for column in columns:
        mean = ts.iloc[train_indexes].mean(numeric_only=True)[column]
        std = ts.iloc[train_indexes].std(numeric_only=True)[column]
        up_border = mean + std_num * std
        down_border = mean - std_num * std
        ts[column] = ts[column].where(ts[column] <= up_border, up_border)
        ts[column] = ts[column].where(ts[column] >= down_border, down_border)

    bins = 39
    window_days = 5
    for column in columns:
        ts[f'{column}_agg'] = np.NaN
    all_hhmm = pd.unique(ts['hhmm'])

    for i in range(0, ts.shape[0] - bins * (window_days + 1) + 1, 39):
        for hhmm in all_hhmm:
            hhmm_df = ts.iloc[i:bins * window_days + i].loc[ts['hhmm'] == hhmm]
            for column in columns:
                vol_agg = hhmm_df.loc[:, column].mean()
                ts.iloc[i + bins * window_days:i + bins * (window_days + 1)].loc[
                    ts['hhmm'] == hhmm, f'{column}_agg'] = vol_agg

    ts_unseason = ts.copy(deep=True)

    for column in columns:
        ts_unseason[column] = ts_unseason[column] - ts_unseason[f'{column}_agg']

    ts_unseason.dropna(inplace=True)

    # features
    df = ts_unseason[columns + ['hhmm', 'VOLUME_fut_agg']].copy(deep=True)

    target = ['VOLUME_fut']
    features_columns = []
    for column in columns:
        for lag in periods:
            feature_col_name = f'{column}_lag_{lag}'
            df[feature_col_name] = df.shift(lag)[column]
            features_columns.append(feature_col_name)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    cat = CatBoostRegressor(iterations=100)
    cat.fit(df[features_columns], df[target], verbose=0, plot=False)

    explainer = shap.TreeExplainer(cat)
    shap_values = explainer.shap_values(Pool(df[features_columns], df[target]))

    top_features = pd.DataFrame(shap_values, columns=df[features_columns].columns).apply(
        lambda x: abs(x)).sum().sort_values(ascending=False)[:10].index

    # splitting and scaling
    val_size /= (1 - test_size)
    x_train, x_test, y_train, y_test = train_test_split(df[top_features], df[target], test_size=test_size,
                                                        shuffle=False)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, shuffle=False)

    scaler = MinMaxScaler()
    scaler.fit(x_train)

    x_train = pd.DataFrame(data=scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
    x_val = pd.DataFrame(data=scaler.transform(x_val), index=x_val.index, columns=x_val.columns)
    x_test = pd.DataFrame(data=scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

    return x_train, x_val, x_test, y_train, y_val, y_test, df


def init_fit(x_train, x_val, y_train, y_val, model_type='CatBoost'):
    param_grid = {'depth': [3], 'l2_leaf_reg': [1], 'learning_rate': [0.09], 'models_num': [10],
                  'early_stopping_rounds': [30]}
    for idx, params in enumerate(ParameterGrid(param_grid)):
        models_num = params['models_num']
        cv_dataset = Pool(data=pd.concat([x_train, x_val]), label=pd.concat([y_train, y_val]))

        cat_params = {"iterations": 500,
                      "depth": params['depth'],
                      "l2_leaf_reg": params['l2_leaf_reg'],
                      "learning_rate": params['learning_rate'],
                      "loss_function": "MAE",
                      "custom_metric": 'R2',
                      # "eval_metric": 'BalancedAccuracy',
                      "use_best_model": True,
                      "verbose": False}

        res = cat_cv(cv_dataset,
                     cat_params,
                     fold_count=models_num,
                     shuffle=False,
                     early_stopping_rounds=params['early_stopping_rounds'],
                     type='TimeSeries ',
                     return_models=True,
                     plot=False,
                     logging_level='Silent')
    ensemble = res[1]
    return ensemble


def predict_evaluate(x_test, df, ensemble):
    models_num = len(ensemble)
    cat = sum_models(ensemble, [1 / models_num] * models_num)

    prediction_cat = pd.DataFrame(cat.predict(x_test), index=x_test.index, columns=['prediction'])

    tdf_test = prediction_cat.join(df)[['prediction', 'VOLUME_fut_agg', 'VOLUME_fut']].copy(deep=True)
    tdf_test['VOLUME_prediction'] = tdf_test['prediction'] + tdf_test['VOLUME_fut_agg']
    tdf_test['VOLUME_original'] = tdf_test['VOLUME_fut'] + tdf_test['VOLUME_fut_agg']

    std_orig = tdf_test.std(numeric_only=True)['VOLUME_original']
    r2_cat = r2_score(tdf_test['VOLUME_original'], tdf_test['VOLUME_prediction'])
    mae_cat = mean_absolute_error(tdf_test['VOLUME_original'], tdf_test['VOLUME_prediction'])
    mape_cat = mean_absolute_percentage_error(tdf_test['VOLUME_original'], tdf_test['VOLUME_prediction'])

    res = {'r2': r2_cat, 'mae': mae_cat, 'mape': mape_cat, 'mae/std': mae_cat / std_orig}
    return res, tdf_test['VOLUME_original'], tdf_test['VOLUME_prediction']
