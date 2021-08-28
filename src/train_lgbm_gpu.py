import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('max_rows', 300)
pd.set_option('max_columns', 300)
import glob
import lightgbm as lgbm
from sklearn.model_selection import KFold
import pickle
import datetime

import os
import sys
sys.path.append('..')

from utils import calc_wap, calc_wap2, log_return, realized_volatility, count_unique, calc_mean_importance, calc_model_importance, plot_importance, rmspe, feval_RMSPE
from preprocess import create_all_feature

LOAD_DATA = True

def main():
    dt = datetime.date.today()
    fm_path = "../output/feature_model/" + dt.strftime("%Y%m%d")
    if not os.path.isdir(fm_path):
        num = 0
    else:
        num = sum(os.path.isdir(os.path.join(fm_path, name)) for name in os.listdir(fm_path))
    fm_path += '/' + str(num)
    os.makedirs(fm_path)
        
    if LOAD_DATA:
        with open('/home/yoshikawa/work/kaggle/OPVP/output/feature_model/20210824/0/train.pkl', 'rb') as f:
            df_train = pickle.load(f)
    else:
        df_train, df_test = create_all_feature()
        
    # 特徴量保存
    pickle.dump(df_train, open(os.path.join(fm_path, "train.pkl"), 'wb'))
    
    X = df_train.drop(['row_id','target'],axis=1)
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
    
    kf = KFold(n_splits=5, random_state=55, shuffle=True)
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
    
    # モデル保存
    for i, model in enumerate(models):
        pickle.dump(model, open(fm_path+"/lgbm"+str(i)+".pkl", 'wb'))
    X_valid.columns.to_series().to_csv(os.path.join(fm_path, "columns.csv"), index=False)

if __name__ == "__main__":
    main()