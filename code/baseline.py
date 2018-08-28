import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import datetime
import lightgbm as lgb
from sklearn.decomposition import KernelPCA

def zuhe(line):
    line['hour'] = line['hour'].astype(str)
    line['weekday'] = line['weekday'].astype(str)
    line['s_ij'] = line['s_ij'].astype(str)
    line['e_ij'] = line['e_ij'].astype(str)

#     line['is_peek'] = line['is_peek'].astype(str)
    line['is_crowd'] = line['is_crowd'].astype(str)
                                   
    line['hour_weekday'] = line['hour'] + line['weekday']
    line['hour_crowd'] = line['hour'] + line['is_crowd']
    line['hour_s'] = line['hour'] + line['s_ij']
    line['hour_e'] = line['hour'] + line['e_ij']

    line['hour'] = line['hour'].astype(int)
#     line['is_peek'] = line['is_peek'].astype(int)
    line['is_crowd'] = line['is_crowd'].astype(int)
    line['weekday'] = line['weekday'].astype(int)
    line['s_ij'] = line['s_ij'].astype(int)
    line['e_ij'] = line['e_ij'].astype(int)
    
    line['hour_weekday'] = line['hour_weekday'].astype(int)
    line['hour_crowd'] = line['hour_crowd'].astype(int)
    line['hour_s'] = line['hour_s'].astype(int)
    line['hour_e'] = line['hour_e'].astype(int)
    return line

with open('./data/train-id4-crowd-grid3.txt', 'rb') as data_file:
    line = pickle.load(data_file)
    line = zuhe(line)
line = line[(line['Diff_Time']<600)]
line['ID'] = line['ID'].astype('category')
line['s_ij'] = line['s_ij'].astype('category') 
line['e_ij'] = line['e_ij'].astype('category')
print('read train finished........')

train = line

col = [c for c in train if
       c not in ['Unnamed: 0','ID2','s_x', 's_y', 'e_x', 'e_y','O_LINENO', 'O_UP', 'Source_Station', 'Target_Station','O_TIME', 'aver_v','max_v', 'Diff_Time']]

col = [c for c in line if
       c not in ['ID','Unnamed: 0','ID2','s_x', 's_y', 'e_x', 'e_y', 'Source_Station', 'Target_Station','O_TIME', 'aver_v','max_v', 'Diff_Time']]

X = train[col].values
y = train['Diff_Time'].values
y_norm = y / 600

print('finished create X,y ................ ')


with open('./data/test-id4-crowd-grid3.txt', 'rb') as data_file:
    test = pickle.load(data_file)
    
col1 = [c for c in test if
       c not in ['Unnamed: 0','ID2' , 's_x', 's_y', 'e_x', 'e_y','O_LINENO', 'O_UP', 'Source_Station', 'Target_Station', 'O_TIME', 'aver_v', 'max_v',
                 'Diff_Time','Distance1', 'distance2','TERMINALNO', 'new_dist']]

col1 = [c for c in test if
       c not in ['ID', 'Unnamed: 0','ID2' , 's_x', 's_y', 'e_x', 'e_y','Source_Station', 'Target_Station', 'O_TIME', 'aver_v', 'max_v',
                 'Diff_Time','Distance1', 'distance2','TERMINALNO', 'new_dist']]

X_test = test[col1].values

print('finished loading X_test ................ ')


def train2(X,y,X_test,n_folds):
    
    def mse_v2(y_pred, train_data):
        y_true = train_data.get_label()
        return 'rmse', 600*(np.mean((y_pred-y_true)**2))**0.5, False
        
    from sklearn.model_selection import StratifiedKFold, KFold
    print("folding")
    kf = KFold(n_splits=n_folds,shuffle=True,random_state=2018)
    result = np.zeros((len(X_test), 1))
    print("training")
    count = 0
    for (tr_idx, val_idx) in kf.split(y):
        X_train = X[tr_idx]
        y_train = y[tr_idx]

        X_dev = X[val_idx]
        y_dev = y[val_idx]
        
        print(X_train.shape)
        print(X_dev.shape)
        
        lgb_train  = lgb.Dataset(X_train, y_train,)#feature_name=col,categorical_feature=['ID']
        lgb_eval = lgb.Dataset(X_dev, y_dev,reference=lgb_train)
        params = {'num_leaves':60, 
                  'max_depth':8,
                  'seed':2018,
                  'colsample_bytree':0.8,
                  'subsample':0.9,
                  'num_threads':20,
                  'n_estimators':20000,
                  # 'learning_rate': 0.1,
                  'objective':'regression_l2',  
                  # 'objective': 'xentropy',
                  'metric':'rmse',
                'device_type':'gpu',
                 }
        gbm = lgb.train(params, 
                        lgb_train,
                        early_stopping_rounds=200,
                        valid_sets=lgb_eval,
                        verbose_eval=50,
                        learning_rates=lambda iter: 0.05 if iter > 15000 else 0.1,
                        # feval=mse_v2,
                       )

        resultx = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        resultx = np.reshape(resultx,(490380, 1))
        print(resultx.shape,result.shape)
        result = result + resultx
        y_te_pred = gbm.predict(X_dev, num_iteration=gbm.best_iteration)
        #print(log_loss(y_dev, y_te_pred))

        count = count+1
    # 提交结果
    result /= n_folds
    return result

pred = train2(X,y,X_test,n_folds=10)

with open('result_0802_without_ID_UP.txt','wb') as f:
    pickle.dump(pred,f)


    
# 29731 l2
# 29700 x



