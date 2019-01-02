import sys
import datetime
import numpy as np
import pandas as pd
from pandas import Series
from itertools import groupby


assert len(sys.argv) == 3

df = pd.read_csv(sys.argv[1])
ids = df['KPI ID'].unique()

# train set
dfs = []
for _id in ids:
    df_tmp = df[df['KPI ID'] == _id]
    df_tmp['datetime'] = pd.to_datetime(df_tmp['timestamp'], unit='s')
    df_tmp = df_tmp.set_index('datetime')
    df_tmp = df_tmp.sort_index()
    
    df_tmp_value = df_tmp['value'].resample('1T').interpolate()
    if 'label' in df.columns: 
    	# train set
    	df_tmp_label = df_tmp['label'].resample('1T').ffill()
    	df_tmp = pd.concat([df_tmp_value, df_tmp_label], axis=1, join='inner')
    else: 
    	# test set
    	df_tmp = df_tmp_value.to_frame()

    df_tmp.insert(1, 'KPI ID', _id)
    df_tmp['timestamp'] = (df_tmp_value.index.astype('int64') // 1e9).astype('int64')
    
    df_tmp['id'] = df_tmp['KPI ID'] + '-' + df_tmp.index.astype('str')
    df_tmp = df_tmp.set_index('id')
    dfs.append(df_tmp)

df_upsampled = pd.concat(dfs)

# normalization
for _id in ids:
    df_tmp = df_upsampled[df_upsampled['KPI ID'] == _id]
    df_tmp['value_diff'] = df_tmp['value'].diff()
    
    df_upsampled.loc[df_upsampled['KPI ID'] == _id, 'value'] = (df_tmp['value'] - df_tmp['value'].mean()) / df_tmp['value'].std()
    df_upsampled.loc[df_upsampled['KPI ID'] == _id, 'value_diff'] = (df_tmp['value_diff'] - df_tmp['value_diff'].mean()) / df_tmp['value_diff'].std()

df_upsampled.loc[df_upsampled['value_diff'].isnull(), 'value_diff'] = 0.0

# save preprocessed data to csv
df_upsampled.to_csv(sys.argv[2])