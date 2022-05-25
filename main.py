from utils.funcs import prepare_data, init_fit, predict_evaluate


x_tr, x_vl, x_tt, y_tr, y_vl, y_tt, df = prepare_data('./data/fut_nq.csv',
                                                      etf_file_path='./data/etf_nq.csv',
                                                      opt_file_path='./data/opt_nq.csv',
                                                      periods=(1, 2, 3, 4, 37, 38, 39, 40))

ensemble = init_fit(x_tr, x_vl, y_tr, y_vl, model_type='CatBoost')

result, vol_orig, vol_pred = predict_evaluate(x_tt, df, ensemble)

print(result)
