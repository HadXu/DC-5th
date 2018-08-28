from keras.models import Sequential
from keras.layers import Dense, Activation
import pickle
from keras.callbacks import EarlyStopping
import keras.backend as K
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
earlyStopping=EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')


with open('./data/train-id.txt', 'rb') as data_file:
    train = pickle.load(data_file)
from sklearn import preprocessing
x1 = train['ID'].as_matrix().reshape([-1,1])
N1 = x1.shape[0]
with open('./data/test3-id.txt', 'rb') as data_file:
    test = pickle.load(data_file)
x2 = test['ID'].as_matrix().reshape([-1,1])
N2 = x2.shape[0]
x = np.concatenate((x1,x2))
x = preprocessing.LabelEncoder().fit_transform(x)
train['ID'] = x[:N1]
test['ID'] = x[N1:]

col = [c for c in train if
       c not in ['O_LINENO', 'O_UP', 'Source_Station', 'Target_Station', 'O_TIME', 'aver_v', 'max_v',
                 'Diff_Time']]
X_train = train[col].values
y_train = train['Diff_Time'].values
X_test = test[col].values
print(X_train.shape, y_train.shape)
print(X_test.shape)
# X_dev = dev[col].values
# y_dev = dev['Diff_Time'].values
# print(X_dev.shape, y_dev.shape)


model = Sequential()
model.add(Dense(units=32, input_dim=14))
model.add(Activation("relu"))
model.add(Dense(units=64))
model.add(Activation("relu"))
model.add(Dense(units=1))
model.add(Activation("relu"))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=[root_mean_squared_error])
model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=1024,callbacks=[earlyStopping])
model.save("keras.h5")

y_pred =  model.predict(X_test,batch_size=1024)[:,0]

test['pred'] = y_pred
sub1 = test[['O_LINENO','TERMINALNO', 'O_UP','Source_Station','Target_Station','O_TIME','pred']]
sub1.columns = ['LINE','TERMINALNO','UP','pred_start_stop_ID','pred_end_stop_ID','realTime','pred']
sub1 = sub1.reset_index()
del sub1['index']

sub=pd.read_csv("./toBePredicted_0601.csv", sep=",")
sub['realTime'] = pd.to_datetime(sub['realTime'],format='%Y-%m-%d %H:%M:%S')
sub2 = sub[['LINE','TERMINALNO','UP','pred_start_stop_ID','pred_end_stop_ID','realTime']]
sub2=pd.merge(sub2,sub1,on=['LINE','TERMINALNO','UP','pred_start_stop_ID','pred_end_stop_ID','realTime'],how='left')
sub2.to_csv('./toBePredicted_0603_result3x.csv',sep=",",index=False)

print('finished')
