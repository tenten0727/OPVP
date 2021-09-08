import pandas as pd
import numpy as np
from utils import calc_wap, calc_wap2, calc_wap3, calc_wap4, log_return, realized_volatility, count_unique, ffill
from sklearn.model_selection import KFold
from sklearn import manifold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


data_dir = '../input/optiver-realized-volatility-prediction/'
last_seconds = [100, 200, 300, 400, 500]

nnn = ['time_id',
     'log_return_realized_volatility_0c1',
     'log_return_realized_volatility_1c1',     
     'log_return_realized_volatility_2c1',
     'log_return_realized_volatility_3c1',     
     'log_return_realized_volatility_6c1',
     'total_volume_mean_0c1',
     'total_volume_mean_1c1', 
     'total_volume_mean_2c1',
     'total_volume_mean_3c1', 
     'total_volume_mean_6c1',
     'trade_size_mean_0c1',
     'trade_size_mean_1c1', 
     'trade_size_mean_2c1',
     'trade_size_mean_3c1', 
     'trade_size_mean_6c1',
     'trade_order_count_mean_0c1',
     'trade_order_count_mean_1c1',
     'trade_order_count_mean_2c1',
     'trade_order_count_mean_3c1',
     'trade_order_count_mean_6c1',      
     'price_spread_mean_0c1',
     'price_spread_mean_1c1',
     'price_spread_mean_2c1',
     'price_spread_mean_3c1',
     'price_spread_mean_6c1',   
     'bid_spread_mean_0c1',
     'bid_spread_mean_1c1',
     'bid_spread_mean_2c1',
     'bid_spread_mean_3c1',
     'bid_spread_mean_6c1',       
     'ask_spread_mean_0c1',
     'ask_spread_mean_1c1',
     'ask_spread_mean_2c1',
     'ask_spread_mean_3c1',
     'ask_spread_mean_6c1',   
     'volume_imbalance_mean_0c1',
     'volume_imbalance_mean_1c1',
     'volume_imbalance_mean_2c1',
     'volume_imbalance_mean_3c1',
     'volume_imbalance_mean_6c1',
     'size_tau2_0c1',
     'size_tau2_1c1',
     'size_tau2_2c1',
     'size_tau2_3c1',
     'size_tau2_6c1'
     ] 

def preprocessor_book(file_path):
    df = pd.read_parquet(file_path)
    # df = ffill(df)
    df['wap'] = calc_wap(df)
    df['log_return'] = df.groupby('time_id')['wap'].apply(log_return)
    df['wap2'] = calc_wap2(df)
    df['log_return2'] = df.groupby('time_id')['wap2'].apply(log_return)
    df['wap3'] = calc_wap3(df)
    df['log_return3'] = df.groupby('time_id')['wap3'].apply(log_return)
    df['wap4'] = calc_wap4(df)
    df['log_return4'] = df.groupby('time_id')['wap4'].apply(log_return)

    df['wap_balance'] = abs(df['wap'] - df['wap2'])
    df['wap_balance2'] = abs(df['wap3'] - df['wap4'])
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1'])/2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2'])/2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df['total_volume'] = df['ask_size1'] + df['bid_size1'] + df['ask_size2'] + df['bid_size2']
    df['ask_volume'] = df['ask_size1'] + df['ask_size2']
    df['bid_volume'] = df['bid_size1'] + df['bid_size2']
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    
    create_feature_dict = {
        'log_return':[realized_volatility, np.mean, np.std],
        'log_return2':[realized_volatility, np.mean, np.std],
        'log_return3':[realized_volatility, np.mean, np.std],
        'log_return4':[realized_volatility, np.mean, np.std],
        'wap_balance':[np.mean, np.std, np.max],
        'wap_balance2':[np.mean, np.std, np.max],
        'price_spread':[np.mean, np.std, np.max],
        'price_spread2':[np.mean, np.std, np.max],
        'bid_spread':[np.mean, np.std, np.max],
        'ask_spread':[np.mean, np.std, np.max],
        'volume_imbalance':[np.mean, np.std, np.max],
        'total_volume':[np.mean, np.std, np.max],
        'ask_volume':[np.mean, np.std, np.max],
        'bid_volume':[np.mean, np.std, np.max],
        'wap':[np.mean, np.std],
        'wap2':[np.mean, np.std],
        'wap3':[np.mean, np.std],
        'wap4':[np.mean, np.std],
    }
    
    df_feature = pd.DataFrame(df.groupby(['time_id']).agg(create_feature_dict)).reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        
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
    df['seconds_diff'] = df.groupby('time_id')['seconds_in_bucket'].apply(lambda x: x.diff())
    df['amount'] = df['price'] * df['size']
    df['amount_seconds_diff'] = df['amount'] * df['seconds_diff']
    df['log_return_per_size'] = df['log_return'] / df['size']
    df['log_return_per_amount'] = df['log_return'] / df['amount']
    df['log_return_seconds_diff'] = df['log_return'] * df['seconds_diff']
    # print(df.head())
    aggregate_dictionary = {
        'log_return':[realized_volatility],
        'log_return_per_size':[realized_volatility],
        'log_return_per_amount':[realized_volatility],
        'log_return_seconds_diff':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.mean, np.max, np.min, np.sum],
        'order_count':[np.mean, np.max, np.sum],
        'amount':[np.mean, np.max, np.min, np.sum],
        'amount_seconds_diff':[np.mean, np.max, np.min],
    }
    
    df_feature = df.groupby('time_id').agg(aggregate_dictionary)
    df_feature = df_feature.reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]
    
    # 値動きの大きさ的なものを表してる？
    def tendency(price, vol):
        df_diff = np.diff(price)
        val = (df_diff/price[1:]) * 100
        power = np.sum(val*vol[1:])
        return (power)

    lis = []
    for n_time_id in df['time_id'].unique():
        df_id = df[df['time_id'] == n_time_id]
        tendencyV = tendency(df_id['price'].values, df_id['size'].values)
        f_max = np.sum(df_id['price'].values > np.mean(df_id['price'].values))
        f_min = np.sum(df_id['price'].values < np.mean(df_id['price'].values))
        df_max = np.sum(np.diff(df_id['price'].values) > 0)
        df_min = np.sum(np.diff(df_id['price'].values) < 0)
        
        abs_diff = np.median(np.abs(df_id['price'].values - np.mean(df_id['price'].values)))
        energy = np.mean(df_id['price'].values ** 2)
        # 四分位範囲
        iqr_p = np.percentile(df_id['price'].values, 75) - np.percentile(df_id['price'].values, 25)
        
        abs_diff_v = np.median(np.abs(df_id['size'].values - np.mean(df_id['size'].values)))
        energy_v = np.mean(df_id['size'].values ** 2)
        iqr_p_v = np.percentile(df_id['size'].values, 75) - np.percentile(df_id['size'].values, 25)
        
        lis.append({'time_id':n_time_id,'tendency':tendencyV,'f_max':f_max,'f_min':f_min,'df_max':df_max,'df_min':df_min,
            'abs_diff':abs_diff,'energy':energy,'iqr_p':iqr_p,'abs_diff_v':abs_diff_v,'energy_v':energy_v,'iqr_p_v':iqr_p_v})
        
    df_lr = pd.DataFrame(lis)
    df_feature = df_feature.merge(df_lr, how = 'left', left_on = 'time_id_', right_on = 'time_id')

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

def add_volatility_per_volume(df_train, df_test):
    df_train['trade_volatility_per_volume'] = df_train['trade_log_return_realized_volatility'] / df_train['trade_order_count_mean']
    df_train['volatility_per_volume'] = df_train['log_return_realized_volatility'] / df_train['total_volume_mean']
    df_train['volatility2_per_volume'] = df_train['log_return2_realized_volatility'] / df_train['total_volume_mean']
    df_train['volatility3_per_volume'] = df_train['log_return3_realized_volatility'] / df_train['total_volume_mean']
    df_train['volatility4_per_volume'] = df_train['log_return4_realized_volatility'] / df_train['total_volume_mean']

    df_test['trade_volatility_per_volume'] = df_test['trade_log_return_realized_volatility'] / df_test['trade_order_count_mean']
    df_test['volatility_per_volume'] = df_test['log_return_realized_volatility'] / df_test['total_volume_mean']
    df_test['volatility2_per_volume'] = df_test['log_return2_realized_volatility'] / df_test['total_volume_mean']
    df_test['volatility3_per_volume'] = df_test['log_return3_realized_volatility'] / df_test['total_volume_mean']
    df_test['volatility4_per_volume'] = df_test['log_return4_realized_volatility'] / df_test['total_volume_mean']

    for i in last_seconds:
        df_train['trade_volatility_per_volume_'+str(i)] = df_train['trade_log_return_realized_volatility_'+str(i)] / df_train['trade_order_count_mean_'+str(i)]
        df_train['volatility_per_volume_'+str(i)] = df_train['log_return_realized_volatility_'+str(i)] / df_train['total_volume_mean_'+str(i)]
        df_train['volatility2_per_volume_'+str(i)] = df_train['log_return2_realized_volatility_'+str(i)] / df_train['total_volume_mean_'+str(i)]
        df_train['volatility3_per_volume_'+str(i)] = df_train['log_return3_realized_volatility_'+str(i)] / df_train['total_volume_mean_'+str(i)]
        df_train['volatility4_per_volume_'+str(i)] = df_train['log_return4_realized_volatility_'+str(i)] / df_train['total_volume_mean_'+str(i)]

        df_test['trade_volatility_per_volume_'+str(i)] = df_test['trade_log_return_realized_volatility_'+str(i)] / df_test['trade_order_count_mean_'+str(i)]
        df_test['volatility_per_volume_'+str(i)] = df_test['log_return_realized_volatility_'+str(i)] / df_test['total_volume_mean_'+str(i)]
        df_test['volatility2_per_volume_'+str(i)] = df_test['log_return2_realized_volatility_'+str(i)] / df_test['total_volume_mean_'+str(i)]
        df_test['volatility3_per_volume_'+str(i)] = df_test['log_return3_realized_volatility_'+str(i)] / df_test['total_volume_mean_'+str(i)]
        df_test['volatility4_per_volume_'+str(i)] = df_test['log_return4_realized_volatility_'+str(i)] / df_test['total_volume_mean_'+str(i)]

    return df_train, df_test

def add_relative_distance(df_train, df_test):
    mean_stock_col = [col for col in list(df_train) if 'mean_stock' in col]
    for c in mean_stock_col:
        col_name = c.replace('_mean_stock', '')
        df_train[col_name+'_relative'] = df_train[col_name] - df_train[c]
        df_test[col_name+'_relative'] = df_test[col_name] - df_test[c]
    
    mean_time_col = [col for col in list(df_train) if 'mean_time' in col]
    for c in mean_time_col:
        col_name = c.replace('_mean_time', '')
        df_train[col_name+'_relative'] = df_train[col_name] - df_train[c]
        df_test[col_name+'_relative'] = df_test[col_name] - df_test[c]

    return df_train, df_test

def add_feature_tau(df_train, df_test):
    # 単位時間あたりの注文数、エントリー数
    df_train['size_tau'] = np.sqrt(1/df_train['trade_seconds_in_bucket_count_unique'])
    df_test['size_tau'] = np.sqrt(1/df_test['trade_seconds_in_bucket_count_unique'])
    df_train['size_tau_400'] = np.sqrt(1/df_train['trade_seconds_in_bucket_count_unique_400'])
    df_test['size_tau_400'] = np.sqrt(1/df_test['trade_seconds_in_bucket_count_unique_400'])
    df_train['size_tau_300'] = np.sqrt(1/df_train['trade_seconds_in_bucket_count_unique_300'])
    df_test['size_tau_300'] = np.sqrt(1/df_test['trade_seconds_in_bucket_count_unique_300'])
    df_train['size_tau_200'] = np.sqrt(1/df_train['trade_seconds_in_bucket_count_unique_200'])
    df_test['size_tau_200'] = np.sqrt(1/df_test['trade_seconds_in_bucket_count_unique_200'])
    
    # tau2 
    df_train['size_tau2'] = np.sqrt(1/df_train['trade_order_count_sum'])
    df_test['size_tau2'] = np.sqrt(1/df_test['trade_order_count_sum'])
    df_train['size_tau2_400'] = np.sqrt(0.25/df_train['trade_order_count_sum'])
    df_test['size_tau2_400'] = np.sqrt(0.25/df_test['trade_order_count_sum'])
    df_train['size_tau2_300'] = np.sqrt(0.5/df_train['trade_order_count_sum'])
    df_test['size_tau2_300'] = np.sqrt(0.5/df_test['trade_order_count_sum'])
    df_train['size_tau2_200'] = np.sqrt(0.75/df_train['trade_order_count_sum'])
    df_test['size_tau2_200'] = np.sqrt(0.75/df_test['trade_order_count_sum'])

    # delta tau
    df_train['size_tau2_d'] = df_train['size_tau2_400'] - df_train['size_tau2']
    df_test['size_tau2_d'] = df_test['size_tau2_400'] - df_test['size_tau2']
    return df_train, df_test

def add_feature_pca(df_train, df_test):
    train_num_data = df_train.drop(['stock_id', 'time_id', 'row_id', 'target'], axis=1)
    test_num_data = df_test.drop(['stock_id', 'time_id', 'row_id'], axis=1)
    train_num_data.replace([np.inf, -np.inf], np.nan, inplace = True)
    test_num_data.replace([np.inf, -np.inf], np.nan, inplace = True)
    train_num_data = train_num_data.fillna(train_num_data.mean())
    test_num_data = test_num_data.fillna(train_num_data.mean())

    scaler = StandardScaler()
    train_standard = scaler.fit_transform(train_num_data)
    test_standard = scaler.transform(test_num_data)
    
    # tsne = manifold.TSNE(n_components=2, random_state=55)
    # train_tsne = tsne.fit_transform(train_scaler)
    # test_tsne = tsne.transform(test_scaler)
    
    pca = PCA(n_components=30)
    train_pca = pca.fit_transform(train_standard)
    test_pca = pca.transform(test_standard)
    df_train_pca = pd.DataFrame(train_pca).add_prefix('pca_')
    df_test_pca = pd.DataFrame(test_pca).add_prefix('pca_')
    
    return pd.concat([df_train, df_train_pca], axis=1), pd.concat([df_test, df_test_pca], axis=1)
    
def add_cluster_feature(df_train, df_test):
    col_names = list(df_train)
    col_names.remove('target')
    col_names.remove('row_id')
    col_names.remove('stock_id')
    col_names.remove('time_id')

    train = df_train.replace([np.inf, -np.inf], np.nan)
    test = df_test.replace([np.inf, -np.inf], np.nan)

    for col in col_names:
        #正規分布になるよう非線形変換
        qt = QuantileTransformer(random_state=55, n_quantiles=2000, output_distribution='normal')
        train[col] = qt.fit_transform(train[[col]]) #seriesかdataframeかで次元が変わる
        test[col] = qt.transform(test[[col]])
    
    train_p = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
    train_p = train_p.pivot(index='time_id', columns='stock_id', values='target')

    corr = train_p.corr()
    ids = corr.index
    kmeans = KMeans(n_clusters = 7, random_state=55).fit(corr.values)

    l = []
    for n in range(7):
        l.append([(x-1) for x in ((ids + 1) * (kmeans.labels_ == n)) if x > 0])
        
    mat = []
    matTest = []

    for n, ind in enumerate(l):
        df_new = train.loc[train['stock_id'].isin(ind)]
        df_new = df_new.groupby(['time_id']).agg(np.nanmean)
        df_new.loc[:, 'stock_id'] = str(n) + 'c1'
        mat.append(df_new)

        df_new = test.loc[test['stock_id'].isin(ind) ]    
        df_new = df_new.groupby(['time_id']).agg(np.nanmean)
        df_new.loc[:,'stock_id'] = str(n) + 'c1'
        matTest.append(df_new)

    mat1 = pd.concat(mat).reset_index()
    mat1.drop(columns=['target'], inplace=True)
    mat2 = pd.concat(matTest).reset_index()

    matTest = []
    mat = []
    kmeans = []

    mat2 = pd.concat([mat2,mat1.loc[mat1.time_id==5]])
    mat1 = mat1.pivot(index='time_id', columns='stock_id')
    mat1.columns = ["_".join(x) for x in mat1.columns.ravel()]
    mat1.reset_index(inplace=True)

    mat2 = mat2.pivot(index='time_id', columns='stock_id')
    mat2.columns = ["_".join(x) for x in mat2.columns.ravel()]
    mat2.reset_index(inplace=True)

    df_train = pd.merge(df_train,mat1[nnn],how='left',on='time_id')
    df_test = pd.merge(df_test,mat2[nnn],how='left',on='time_id')

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
    
    df_train['time_id'] = df_train['row_id'].apply(lambda x:(x.split('-')[1])).astype(int)
    df_train, df_test = add_volatility_per_volume(df_train, df_test)
    df_train, df_test = add_feature_tau(df_train, df_test)
    df_train = get_time_stock(df_train)
    df_test = get_time_stock(df_test)
    df_train, df_test = add_relative_distance(df_train, df_test)

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
    _, df_test = add_volatility_per_volume(df_train, df_test)
    _, df_test = add_feature_tau(df_train, df_test)
    df_test = get_time_stock(df_test)
    _, df_test = add_relative_distance(df_train, df_test)

    return df_test

def after_create_feature(df_train, df_test):
    df_train, df_test = add_feature_pca(df_train, df_test)
    df_train, df_test = add_cluster_feature(df_train, df_test)
    return df_train, df_test