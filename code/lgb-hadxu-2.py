import pickle
import lightgbm as lgb
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

# with open('./data/train-id5-crowd-grid5.txt', 'rb') as data_file:
#     line = pickle.load(data_file)

# with open('./data/test-id5-crowd-grid5.txt', 'rb') as data_file:
#     test = pickle.load(data_file)
    
def zuhe(line):
    line['hour'] = line['hour'].astype(str)
    line['weekday'] = line['weekday'].astype(str)
    line['s_ij'] = line['s_ij'].astype(str)
    line['e_ij'] = line['e_ij'].astype(str)
    line['new_s_t'] = line['new_s_t'].astype('str')

#     line['is_peek'] = line['is_peek'].astype(str)
    line['is_crowd'] = line['is_crowd'].astype(str)
                                   
    line['hour_weekday'] = line['hour'] + line['weekday']
    line['hour_crowd'] = line['hour'] + line['is_crowd']
    line['hour_s'] = line['hour'] + line['s_ij']
    line['hour_e'] = line['hour'] + line['e_ij']
    line['hour_s_t'] = line['hour'] + line['new_s_t']

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
    line['hour_s_t'] = line['hour_s_t'].astype(int)

    line['ID'] = line['ID'].astype('category')
    line['s_ij'] = line['s_ij'].astype('category') 
    line['e_ij'] = line['e_ij'].astype('category')
    line['Source_Station_encode'] = line['Source_Station_encode'].astype('category') 
    line['Target_Station_encode'] = line['Target_Station_encode'].astype('category')
    line['new_s_t'] = line['new_s_t'].astype('category')
    line['new_s_t_o'] = line['new_s_t_o'].astype('category')
    return line

with open('./data/train-id5-crowd-grid5.txt', 'rb') as data_file:
    line = pickle.load(data_file)
    line = zuhe(line)#13963746
line = line[(line['Diff_Time']<600)]


train = line
col = [c for c in train if
       c not in ['Unnamed: 0','s_t','s_t_o','ID2','s_x', 's_y', 'e_x', 'e_y','O_LINENO', 'O_UP', 'Source_Station', 'Target_Station','O_TIME', 'aver_v','max_v', 'Diff_Time']]
print(col)
X = train[col].values
y = train['Diff_Time'].values
print(X.shape)

with open('./data/test-id5-crowd-grid5.txt', 'rb') as data_file:
    test = pickle.load(data_file)
test = zuhe(test)

col1 = [c for c in test if
       c not in ['Unnamed: 0','s_t','s_t_o','ID2' , 's_x', 's_y', 'e_x', 'e_y','O_LINENO', 'O_UP', 'Source_Station', 'Target_Station', 'O_TIME', 'aver_v', 'max_v', 'Diff_Time','Distance1', 'distance2','TERMINALNO', 'new_dist']]
print(col1)
X_test = test[col1].values
print(X_test.shape)


print('preprocessing finished..................')

def get_stacking(X_train, labels, n_folds=10):
    train_num, test_num = X_train.shape[0], X_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=1001001)
        
    for n_fold,(train_idx, val_idx) in enumerate(folds.split(labels)):
        train = X_train[train_idx]
        train_y = labels[train_idx]

        val = X_train[val_idx]
        val_y = labels[val_idx]
        
        lgb_train  = lgb.Dataset(train, train_y,)#feature_name=col,categorical_feature=['ID']
        lgb_eval = lgb.Dataset(val, val_y,reference=lgb_train)
        params = {'num_leaves':60, 
                  'max_depth':8,
                  'seed':2018,
                  'colsample_bytree':0.8,
                  'subsample':0.9,
                  'num_threads':25,
                  'n_estimators':25000,
                  'learning_rate': 0.1,
                  'objective':'regression_l2',  
                  'metric':'rmse'}
        print('fold',n_fold)
        model = lgb.train(params, lgb_train,early_stopping_rounds=100,valid_sets=lgb_eval,verbose_eval=50)
        second_level_train_set[val_idx] = model.predict(val)
        test_nfolds_sets[:,n_fold] = model.predict(X_test)
        
        feim = pd.Series(model.feature_importance(),index=col)
        feim = feim.sort_values(ascending=False)
        print(feim)
    
    "---save---"
    with open('test_nfolds_sets_gbm_0810.txt', 'wb') as data_file:
        pickle.dump(test_nfolds_sets, data_file)

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)

    result = second_level_test_set
    
    return second_level_train_set, second_level_test_set, result

train_sets = []
test_sets = []
train_set, test_set, result = get_stacking(X, y, n_folds=10)
train_sets.append(train_set)
test_sets.append(test_set)

meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)
print(meta_train.shape, meta_test.shape)


"---save---"
with open('meta_train_lgb_0810.txt', 'wb') as data_file:
    pickle.dump(meta_train, data_file)
with open('meta_test_lgb_0810.txt', 'wb') as data_file:
    pickle.dump(meta_test, data_file)

    
print('done............')






