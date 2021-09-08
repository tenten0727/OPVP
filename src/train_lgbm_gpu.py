import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('max_rows', 300)
pd.set_option('max_columns', 300)
import glob
import lightgbm as lgbm
from sklearn.model_selection import KFold, StratifiedKFold
import pickle
import datetime
import argparse

import os
import sys
sys.path.append('..')

from utils import calc_wap, calc_wap2, log_return, realized_volatility, count_unique, calc_mean_importance, calc_model_importance, plot_importance, rmspe, feval_RMSPE
from preprocess import after_create_feature, create_all_feature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--save_name', type=str, default='lgbm')
    
    opts = parser.parse_args()
    
    dt = datetime.date.today()
    fm_path = "../output/feature_model/" + dt.strftime("%Y%m%d")
    fm_path += '/' + opts.save_name
    if not os.path.isdir(fm_path):
        os.makedirs(fm_path)
        
    if opts.train_path == '':
        print('Create data...')
        df_train, df_test = create_all_feature()
    else:
        print('Load data...')
        with open(opts.train_path+'/train.pkl', 'rb') as f:
            df_train = pickle.load(f)
        with open(opts.train_path+'/test.pkl', 'rb') as f:
            df_test = pickle.load(f)
        
    # 特徴量保存
    pickle.dump(df_train, open(os.path.join(fm_path, "train.pkl"), 'wb'))
    pickle.dump(df_test, open(os.path.join(fm_path, "test.pkl"), 'wb'))
    
    df_train, df_test = after_create_feature(df_train, df_test)

    X = df_train.drop(['row_id', 'target', 'time_id'],axis=1)
    y = df_train['target']
    
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
    
    num_bins = int(np.floor(1+np.log2(df_train.shape[0])))
    target_bins = pd.cut(df_train['target'], bins=num_bins, labels=False)

    kf = StratifiedKFold(n_splits=5, random_state=55, shuffle=True)
    models = []
    scores = 0.0
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, target_bins)):
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
                    categorical_feature = ['stock_id']                
                    )
        
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

        RMSPE = round(rmspe(y_true = y_valid, y_pred = y_pred),3)
        print(f'Performance of the　prediction: , RMSPE: {RMSPE}')

        #keep scores and models
        scores += RMSPE / 5
        models.append(model)
        print("*" * 100)
        
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