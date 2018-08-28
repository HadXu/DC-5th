import pandas as pd
import numpy as np
from pandas.tseries.offsets import *
import datetime
import os
import pickle
from tqdm import tqdm, tqdm_pandas
import math

tqdm.pandas()

with open('./data/full_history2', 'rb') as data_file:
    hi = pickle.load(data_file)
hi = hi.groupby(['source_id','target_id','is_rain','is_workday','is_peek']).mean().reset_index()
hi['is_workday'] = hi['is_workday'].astype('int') 
hi['is_peek'] = hi['is_peek'].astype('int') 
hi['is_rain'] = hi['is_rain'].astype('int') 

with open('./data/test-id4-crowd-grid6.txt', 'rb') as data_file:
    train = pickle.load(data_file)

def get_grid_history(s,t,w,p,r):
    b = hi[(hi['source_id']==s)&(hi['target_id']==t)&(hi['is_workday']==w)&(hi['is_peek']==p)&(hi['is_rain']==r)]#
    if b.shape[0] > 0:
        return b.iloc[0]
    else :
        dic = {'Distance':[-1],'Diff_Time':[-1]}
        x = pd.DataFrame(dic)
        return x.iloc[0]
print('start')
tmp = train.progress_apply(lambda x : get_grid_history(int(x['Source_Station_encode']),int(x['Target_Station_encode']),x['is_workday'],x['is_peek'],x['is_rain']), axis=1)
train['encode_aver_diff'] = tmp['Diff_Time']
train['encode_aver_d'] = tmp['Distance']
train

with open('./data/test-id4-crowd-grid7.txt', 'wb') as data_file:
    pickle.dump(train, data_file)