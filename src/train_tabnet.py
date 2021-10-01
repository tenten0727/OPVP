import glob
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy as sc
from joblib import Parallel, delayed
from preprocess import create_all_feature
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
pd.set_option('max_columns', 300)
import datetime
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import RMSPE
np.random.seed(0)

import random
from pathlib import Path

SEED = 55
LOAD_DATA = True

def create_folds(data, num_splits,target):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data[target], bins=num_bins, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=num_splits)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data

def random_seed(SEED):
    
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    random_seed(SEED)


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
            train = pickle.load(f)
    else:
        train, _ = create_all_feature()
        
    # 特徴量保存
    pickle.dump(train, open(os.path.join(fm_path, "train.pkl"), 'wb'))


    for col in train.columns.to_list()[2:]:
        train[col] = train[col].fillna(train[col].mean())
        
    scales = train.drop(['row_id', 'target',"stock_id"], axis = 1).columns.to_list()

    scaler = StandardScaler()
    scaler.fit(train[scales])
    
    pickle.dump(train, open(os.path.join(fm_path, "scaler.pkl"), 'wb'))

    le = LabelEncoder
    le.fit(train["stock_id"])
    train["stock_id"] = le.transform(train["stock_id"])

    train = create_folds(train, 5,"target")
    
    tabnet_params = dict(
        n_d = 32,
        n_a = 32,
        n_steps = 3,
        gamma = 1.3,
        lambda_sparse = 0,
        optimizer_fn = optim.Adam,
        optimizer_params = dict(lr = 1e-2, weight_decay = 1e-5),
        mask_type = "entmax",
        scheduler_params = dict(
            mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
        scheduler_fn = ReduceLROnPlateau,
        seed = 42,
        #verbose = 5,
        cat_dims=[len(le.classes_)], cat_emb_dim=[10], cat_idxs=[-1] # define categorical features
    )
    
    max_epochs = 50
    
    bestscores=[]

    for fold in range(5):

        traindf = train[train["kfold"]!=fold].reset_index(drop=True)
        validdf = train[train["kfold"]==fold].reset_index(drop=True)
        ## Normalization except stock id ; stock id is used as categoral features
        X_train = traindf.drop(['row_id', 'target', "kfold","stock_id"], axis = 1).values
        
        X_train = scaler.transform(X_train)
        X_traindf = pd.DataFrame(X_train)

        X_traindf["stock_id"]=traindf["stock_id"]

        X_train = X_traindf.values
        y_train = traindf['target'].values.reshape(-1, 1)

        # validation is same
        X_valid = validdf.drop(['row_id', 'target', "kfold","stock_id"], axis = 1).values
        X_valid = scaler.transform(X_valid)

        X_validdf = pd.DataFrame(X_valid)

        X_validdf["stock_id"]=validdf["stock_id"]

        X_valid = X_validdf.values
        y_valid = validdf['target'].values.reshape(-1, 1)
        
        # calculate weight
        
        y_weight = 1/np.square(traindf["target"])
        
    
        print("----Fold:{}--------start----".format(str(fold)))

        # initialize random seed

        random_seed(SEED)


        # tabnet model

        clf = TabNetRegressor(**tabnet_params)

        # tabnet training

        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=[RMSPE],
            max_epochs=max_epochs,
            patience=10,
            batch_size=1024*2, virtual_batch_size=128*2,
            num_workers=4,
            drop_last=False,
            weights = y_weight,
            loss_fn=nn.L1Loss()
        )


    # save tabnet model
    saving_path_name = os.path.join(fm_path, "tabnet_model_" + str(fold))
    saved_filepath = clf.save_model(saving_path_name)

    bestscores.append(clf.best_cost)

if __name__ == "__main__":
    main()
