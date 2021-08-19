import pandas as pd
import numpy as np
from utils import calc_wap, calc_wap2, log_return, realized_volatility, count_unique


data_dir = '../input/optiver-realized-volatility-prediction/'

def preprocessor_book(file_path):
    df = pd.read_parquet(file_path)
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
    
    last_seconds = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    
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
    
    last_seconds = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    
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