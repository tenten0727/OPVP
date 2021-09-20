import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('max_rows', 300)
pd.set_option('max_columns', 300)
import glob
import lightgbm as lgbm
from sklearn.model_selection import KFold, GroupKFold
import pickle
import datetime
import argparse

import os
import sys
sys.path.append('..')

from utils import calc_wap, calc_wap2, log_return, realized_volatility, count_unique, calc_mean_importance, calc_model_importance, plot_importance, reduce_mem_usage, rmspe, feval_RMSPE
from preprocess import create_all_feature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--save_name', type=str, default='lgbm')
    parser.add_argument('--debug',  action='store_true')
    
    opts = parser.parse_args()
    
    dt = datetime.date.today()
    fm_path = "../output/feature_model/" + dt.strftime("%Y%m%d")
    fm_path += '/' + opts.save_name
    if not os.path.isdir(fm_path):
        os.makedirs(fm_path)
        
    if opts.train_path == '':
        print('Create data...')
        df_train, df_test = create_all_feature(opts.debug)
        print('Create data finish!')
    else:
        print('Load data...')
        with open(opts.train_path+'/train.pkl', 'rb') as f:
            df_train = pickle.load(f)
        with open(opts.train_path+'/test.pkl', 'rb') as f:
            df_test = pickle.load(f)
        print('Load data finish!')
        
    # 特徴量保存
    pickle.dump(df_train, open(os.path.join(fm_path, "train.pkl"), 'wb'))
    pickle.dump(df_test, open(os.path.join(fm_path, "test.pkl"), 'wb'))
    print('save data!')
    # df_train = reduce_mem_usage(df_train)
    # df_test = reduce_mem_usage(df_test)

    X = df_train.drop(df_train[df_train.stock_id==31].index).drop(['row_id', 'target', 'time_id', 'stock_id'],axis=1).reset_index(drop=True)
    X_31 = df_train[df_train.stock_id == 31].drop(['row_id', 'target', 'time_id', 'stock_id'],axis=1).reset_index(drop=True)
    y = df_train.drop(df_train[df_train.stock_id==31].index)['target'].reset_index(drop=True)
    y_31 = df_train[df_train.stock_id==31]['target'].reset_index(drop=True)

    params = {
        "objective": "rmse", 
        "metric": "rmse", 
        "boosting_type": "gbdt",
        'early_stopping_rounds': 30,
        'learning_rate': 0.01,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'feature_fraction': 0.8, # 利用する特徴量の割合
        'bagging_fraction': 0.8,
        'device':'gpu',
    }
    

    kf = KFold(n_splits=5, random_state=55, shuffle=True)

    group =df_train['time_id']
    models = []
    scores = 0.0
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f'Fold: {fold+1}')
        X_train, y_train = X.loc[trn_idx], y.loc[trn_idx]
        X_valid, y_valid = X.loc[val_idx], y.loc[val_idx]
        
        # RMSPEで最適化を行うため? RMSEの損失関数に1/yi^2の重み付けをすればOK
        # https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/250324
        weights = 1/np.square(y_train)
        lgbm_train = lgbm.Dataset(X_train, y_train, weight=weights)
        
        weights = 1/np.square(y_valid)
        lgbm_valid = lgbm.Dataset(X_valid, y_valid, reference=lgbm_train, weight=weights)
        
        model = lgbm.train(params=params,
                    train_set=lgbm_train,
                    valid_sets=[lgbm_train, lgbm_valid],
                    num_boost_round=5000,         
                    feval=feval_RMSPE,
                    verbose_eval=100,
                    # categorical_feature = ['stock_id']                
                    )
        
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

        RMSPE = round(rmspe(y_true = y_valid, y_pred = y_pred),3)
        print(f'Performance of the　prediction: , RMSPE: {RMSPE}')
        # print('val_stock_id: ', list(df_train.loc[val_idx, 'stock_id'].unique()))
        y_pred_31 = model.predict(X_31, num_iteration=model.best_iteration)

        RMSPE_31 = round(rmspe(y_true = y_31, y_pred = y_pred_31),3)
        print(f'Performance of the　prediction stock_id=31: , RMSPE: {RMSPE_31}')
        #keep scores and models
        scores += RMSPE / 5
        models.append(model)
        print("*" * 100)
        
    if not opts.debug:
        print('Score: ', scores)
        with open(fm_path+'/score.txt', 'w') as f:
            f.write(str(scores))
        # モデル保存
        for i, model in enumerate(models):
            pickle.dump(model, open(fm_path+"/lgbm"+str(i)+".pkl", 'wb'))
            imp = calc_model_importance(model, feature_names=X_train.columns.values.tolist())
            plot_importance(imp.head(30), save_filepath=fm_path+'/imp_top30_'+str(i))
            plot_importance(imp.tail(30), save_filepath=fm_path+'/imp_worst30_'+str(i))

        X_valid.columns.to_series().to_csv(os.path.join(fm_path, "columns.csv"), index=False)

if __name__ == "__main__":
    main()