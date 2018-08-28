from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, Flatten, Input, concatenate
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Nadam
import keras.backend as K
import numpy as np
import pandas as pd
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def zuhe(line):
    line['hour'] = line['hour'].astype(str)
    line['weekday'] = line['weekday'].astype(str)
    line['is_peek'] = line['is_peek'].astype(str)
    line['is_workday'] = line['is_workday'].astype(str)
                                   
    line['hour_weekday'] = line['hour'] + line['weekday']
    line['peak_workday'] = line['is_peek'] + line['is_workday']
    
    line['hour'] = line['hour'].astype(int)
    line['is_peek'] = line['is_peek'].astype(int)
    line['is_workday'] = line['is_workday'].astype(int)
    line['weekday'] = line['weekday'].astype(int)
    
    line['hour_weekday'] = line['hour_weekday'].astype(int)
    line['peak_workday'] = line['peak_workday'].astype(int)
    return line

with open('./data/train-id4-crowd-grid4.txt', 'rb') as data_file:
    train = pickle.load(data_file)
    #训练验证都划分
    #900 29 26 #800 27.9 25.9 #700 27.5 25.3 #27.4 24.8 #26.4 24.08 #23 21 #(90)250 21 19.5 #(80)17. 16.5
    #训练划分 验证不划分
    #250
train = train[(train['Diff_Time']<600)]
train = zuhe(train)
with open('./data/test-id4-crowd-grid4.txt', 'rb') as data_file:
    test = pickle.load(data_file)
test = zuhe(test)
from sklearn import preprocessing
x1 = train['ID'].as_matrix().reshape([-1,1])
N1 = x1.shape[0]

x2 = test['ID'].as_matrix().reshape([-1,1])
N2 = x2.shape[0]
x = np.concatenate((x1,x2))
x = preprocessing.LabelEncoder().fit_transform(x) #13963746x21567
print(x.shape)#(14454126, 21567)
train['new_ID'] = x[:N1]
test['new_ID'] = x[N1:]

col1 = ['new_ID']
col2 = [c for c in train if
       c not in ['Unnamed: 0','ID2', 's_ij', 'e_ij', 's_x', 's_y', 'e_x', 'e_y', 'new_ID','ID','O_LINENO', 'O_UP', 'Source_Station', 'Target_Station', 'O_TIME', 'aver_v', 'max_v',
                 'Diff_Time']]
col3 = ['Source_Station_encode']
col4 = ['Target_Station_encode']

X_train1 = train[col1].values
X_train2 = train[col2].values
X_train3 = train[col3].values
X_train4 = train[col4].values
y_train = train['Diff_Time'].values

print(col1,col2,col3,col4)
print('train',X_train1.shape, X_train2.shape, X_train3.shape, X_train4.shape)

col1 = ['new_ID']
col2 = [c for c in test if
       c not in ['Unnamed: 0','ID2', 's_ij', 'e_ij', 's_x', 's_y', 'e_x', 'e_y', 'new_ID','ID','O_LINENO', 'O_UP', 'Source_Station', 'Target_Station', 'O_TIME', 'aver_v', 'max_v',
                 'Diff_Time','Distance1', 'distance2','TERMINALNO', 'new_dist']]
col3 = ['Source_Station_encode']
col4 = ['Target_Station_encode']

print(col1,col2,col3,col4)
X_test1 = test[col1].values
X_test2 = test[col2].values
X_test3 = test[col3].values
X_test4 = test[col4].values
print('test',X_test1.shape,X_test2.shape,X_test3.shape,X_test4.shape)

from keras.layers import Reshape, Conv2D, MaxPooling2D
def get_model():
    input1 = Input(shape=(1,))
    input2 = Input(shape=(20,))
    input3 = Input(shape=(1,))
    input4 = Input(shape=(1,))

    x1 = Embedding(21064, 128, input_length=1)(input1)
    x1 = Flatten()(x1)
    x1 = BatchNormalization()(x1)
    x2 = Dense(units=128)(input2)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    x3 = Embedding(4609, 48, input_length=1)(input3)
    x3 = BatchNormalization()(x3)
    x3 = Flatten()(x3)
    x4 = Embedding(4609, 48, input_length=1)(input4)
    x4 = BatchNormalization()(x4)
    x4 = Flatten()(x4)

    def dist(x):
        return x[0]*x[1]/(K.sum(x[0]**2,axis=1,keepdims=True)+K.sum(x[1]**2,axis=1,keepdims=True))    
    #     Aggregate
    x3_x4 = Lambda(dist)([x3,x4])

    x = concatenate([x1, x2, x3, x4, x3_x4])
    #CNN
    x = Reshape((20, 20, 1))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
#     x = Dense(512, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation=None)(x)
    model = Model(inputs=[input1, input2, input3, input4], outputs=out)
    print(model.summary())
    # model.load_weights('./log/weights-21.7671.hdf5')
    model.compile(loss=root_mean_squared_error, optimizer=Nadam(), metrics=[root_mean_squared_error])
    return model

from sklearn.model_selection import StratifiedKFold, KFold

def get_stacking(X_train1, X_train2, X_train3, X_train4, labels, n_folds=10):
    train_num, test_num = X_train1.shape[0], X_test1.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=1001001)
        
    for n_fold,(train_idx, val_idx) in enumerate(folds.split(labels)):
        train_1 = X_train1[train_idx]
        train_2 = X_train2[train_idx]
        train_3 = X_train3[train_idx]
        train_4 = X_train4[train_idx]
        train_y = labels[train_idx]

        val_1 = X_train1[val_idx]
        val_2 = X_train2[val_idx]
        val_3 = X_train3[val_idx]
        val_4 = X_train4[val_idx]
        val_y = labels[val_idx]
#         ckpt_path = './logtianchi/'+ model_list[n_fold]
#         print(ckpt_path)
        model = get_model()
        if n_fold == 0:
            print(model.summary())
        print(n_fold)
        
        ckpt_path = './log/cv_'+str(n_fold)+'_weights-{val_loss:.4f}.hdf5'
        model.fit([train_1,train_2,train_3,train_4], train_y, validation_data=([val_1,val_2,val_3,val_4], val_y), 
                  epochs=10, 
                  batch_size=1024,
                  callbacks=[
                      ModelCheckpoint(ckpt_path,monitor='val_loss',verbose=1,save_best_only=True,mode='min'),
                      EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')])
        
        second_level_train_set[val_idx] = model.predict([val_1, val_2,val_3,val_4], batch_size=1024)[:, 0]
        test_nfolds_sets[:,n_fold] = model.predict([X_test1, X_test2, X_test3, X_test4], batch_size=1024)[:, 0]
    
    "---save---"
    with open('./data/test_nfolds_sets3.txt', 'wb') as data_file:
        pickle.dump(test_nfolds_sets, data_file)

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)

    result = second_level_test_set
    
    return second_level_train_set, second_level_test_set, result

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)+1e-15, axis=-1))

# satck
train_sets = []
test_sets = []
train_set, test_set, result = get_stacking(X_train1, X_train2, X_train3, X_train4, y_train, n_folds=10)
train_sets.append(train_set)
test_sets.append(test_set)


meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)
print(meta_train.shape, meta_test.shape)

"---save---"
with open('./data/meta_train_nn3.txt', 'wb') as data_file:
    pickle.dump(meta_train, data_file)
with open('./data/meta_test_nn3.txt', 'wb') as data_file:
    pickle.dump(meta_test, data_file)