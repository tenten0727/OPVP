import pandas as pd
import numpy as np
from utils import calc_wap, calc_wap2, log_return, realized_volatility, count_unique, ffill
from sklearn.model_selection import KFold


data_dir = '../input/optiver-realized-volatility-prediction/'

def preprocessor_book(file_path):
    df = pd.read_parquet(file_path)
    # df = ffill(df)
    df['wap'] = calc_wap(df)
    df['log_return'] = df.groupby('time_id')['wap'].apply(log_return)
    df['wap2'] = calc_wap2(df)
    df['log_return2'] = df.groupby('time_id')['wap2'].apply(log_return)

    df['wap_balance'] = abs(df['wap'] - df['wap2'])
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1'])/2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df['total_volume'] = df['ask_size1'] + df['bid_size1'] + df['ask_size2'] + df['bid_size2']
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    
    create_feature_dict = {
        'log_return':[realized_volatility, np.mean, np.std, np.sum],
        'log_return2':[realized_volatility, np.mean, np.std, np.sum],
        'wap_balance':[np.mean, np.std, np.sum],
        'price_spread':[np.mean, np.std, np.sum],
        'bid_spread':[np.mean, np.std, np.sum],
        'ask_spread':[np.mean, np.std, np.sum],
        'volume_imbalance':[np.mean, np.std, np.sum],
        'total_volume':[np.mean, np.std, np.sum],
        'wap':[np.mean, np.std, np.sum],
    }
    
    df_feature = pd.DataFrame(df.groupby(['time_id']).agg(create_feature_dict)).reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]
    
    last_seconds = [150, 300, 450]
    
    for second in last_seconds:
        second = 600 - second
        
        df_feature_sec = pd.DataFrame(df[df['seconds_in_bucket'] >= second].groupby('time_id').agg(create_feature_dict)).reset_index()
        df_feature_sec.columns = ['_'.join(col) for col in df_feature_sec.columns]
        df_feature_sec = df_feature_sec.add_suffix('_' + str(second))
        
        df_feature = pd.merge(df_feature, df_feature_sec, how='left', left_on='time_id_', right_on=f'time_id__{second}')
        df_feature = df_feature.drop([f'time_id__{second}'], axis=1)
        
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature = df_feature.drop(['time_id_'], axis=1)

    return df_feature


def preprocessor_trade(file_path):
    df = pd.read_parquet(file_path)
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    
    aggregate_dictionary = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum],
        'order_count':[np.mean],
    }
    
    df_feature = df.groupby('time_id').agg(aggregate_dictionary)
    df_feature = df_feature.reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]
    
    last_seconds = [150, 300, 450]
    
    for second in last_seconds:
        second = 600 - second
        
        df_feature_sec = pd.DataFrame(df[df['seconds_in_bucket'] >= second].groupby('time_id').agg(aggregate_dictionary)).reset_index()
        df_feature_sec.columns = ['_'.join(col) for col in df_feature_sec.columns]
        df_feature_sec = df_feature_sec.add_suffix('_' + str(second))
        
        df_feature = pd.merge(df_feature, df_feature_sec, how='left', left_on='time_id_', right_on=f'time_id__{second}')
        df_feature = df_feature.drop([f'time_id__{second}'], axis=1)
    
    df_feature = df_feature.add_prefix('trade_')
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature = df_feature.drop(['trade_time_id_'], axis=1)
    
    return df_feature


def preprocessor(list_stock_ids, is_train = True):
    from joblib import Parallel, delayed #並列処理
    df = pd.DataFrame()
    
    def for_joblib(stock_id):
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id) 
        else:
            file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)  
            
        df_tmp = pd.merge(preprocessor_book(file_path_book), preprocessor_trade(file_path_trade), on='row_id', how='left')
        
        return pd.concat([df, df_tmp])
    
    df = Parallel(n_jobs=-1, verbose=1)(
        delayed(for_joblib)(stock_id) for stock_id in list_stock_ids
    )
    
    df = pd.concat(df, ignore_index=True)
    return df

# refs: https://www.kaggle.com/ragnar123/optiver-realized-volatility-lgbm-baseline
def get_time_stock(df):
    vol_cols = [s for s in list(df.columns) if 'realized_volatility' in s]
    
    # Group by the stock id
    df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    # Rename columns joining suffix
    df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix('_' + 'stock')

    # Group by the time id
    df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    # Rename columns joining suffix
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix('_' + 'time')
    
    # Merge with original dataframe
    df = df.merge(df_stock_id, how = 'left', left_on = ['stock_id'], right_on = ['stock_id__stock'])
    df = df.merge(df_time_id, how = 'left', left_on = ['time_id'], right_on = ['time_id__time'])
    df.drop(['stock_id__stock', 'time_id__time'], axis = 1, inplace = True)
    return df

def target_encoding(df_train, df_test, is_test=False):
    df_train['stock_id'] = df_train['row_id'].apply(lambda x:x.split('-')[0])
    df_test['stock_id'] = df_test['row_id'].apply(lambda x:x.split('-')[0])

    stock_id_target_mean = df_train.groupby('stock_id')['target'].mean()
    df_test['stock_id_target_enc'] = df_test['stock_id'].map(stock_id_target_mean)

    if not is_test:
        tmp = np.repeat(np.nan, df_train.shape[0])
        kf = KFold(n_splits=10, shuffle=True, random_state=55)
        for idx_1, idx_2 in kf.split(df_train):
            target_mean = df_train.iloc[idx_1].groupby('stock_id')['target'].mean()

            tmp[idx_2] = df_train['stock_id'].iloc[idx_2].map(target_mean)
        df_train['stock_id_target_enc'] = tmp
    
    return df_train, df_test

def create_all_feature():
    train = pd.read_csv(data_dir + 'train.csv')
    train_ids = train.stock_id.unique()
    df_train = preprocessor(list_stock_ids=train_ids, is_train=True)
    train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
    train = train[['row_id', 'target']]
    df_train = train.merge(df_train, on=['row_id'], how='left')
    
    test = pd.read_csv(data_dir + 'test.csv')
    test_ids = test.stock_id.unique()
    df_test = preprocessor(list_stock_ids= test_ids, is_train = False)
    df_test = test.merge(df_test, on = ['row_id'], how = 'left')
    
    #TE
    df_train, df_test = target_encoding(df_train, df_test)
    
    df_train['time_id'] = df_train['row_id'].apply(lambda x:x.split('-')[1])
    df_train = get_time_stock(df_train)
    df_test = get_time_stock(df_test)
    df_train = df_train.drop(['time_id'], axis=1)
    
    df_train['stock_id'] = df_train['stock_id'].astype(int)
    df_test['stock_id'] = df_test['stock_id'].astype(int)
    
    return df_train, df_test

def create_test_feature(df_train):
    test = pd.read_csv(data_dir + 'test.csv')
    test_ids = test.stock_id.unique()
    df_test = preprocessor(list_stock_ids= test_ids, is_train = False)
    df_test = test.merge(df_test, on = ['row_id'], how = 'left')

    target_encoding(df_train, df_test, is_test=True)

    df_test['stock_id'] = df_test['stock_id'].astype(int)
    df_test = get_time_stock(df_test)
    
    return df_test
