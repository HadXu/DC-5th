# -*- coding: utf-8 -*-
"""
Created on Mon May 21 23:10:48 2018

@author: T
"""

import pandas as pd
import numpy as np
from scipy.stats import hmean

sub = pd.read_csv("./data/toBePredicted_forUser.csv",encoding="gbk")
data = pd.read_csv("toBePredicted_0607.csv", sep=",")

myresult1 = pd.read_csv("qrf_toBePredicted_0807_result.csv", sep=",")
myresult2 = pd.read_csv("toBePredicted_0808_result_lgb_new.csv", sep=",")
# myresult3 = pd.read_csv("toBePredicted_0813y.csv", sep=",")

#lgb toBePredicted_0808_result_lgb_new
#nn qrf_toBePredicted_0807_result

y_pred1 = myresult1['new_pred'].as_matrix()
y_pred2 = myresult2['new_pred'].as_matrix()
# y_pred3 = myresult3['new_pred'].as_matrix()

print(y_pred1.shape)


y_pred1[y_pred1<=0]=0.00000001
y_pred2[y_pred2<=0]=0.00000001
# y_pred3[y_pred3<=0]=0.00000001

# y_pred = y_pred1#hmean([y_pred1, y_pred2])
y = np.concatenate([y_pred1.reshape(-1,1),y_pred2.reshape(-1,1)],axis=1)
y_pred = np.mean(y,axis=1)
# y_pred = hmean([y_pred3, y_pred4])
print(np.max(y_pred),np.min(y_pred))
MAX = 600#650
MIN = 52#52
count = 0 #pred 全局指针
time_all = []
for i in range(len(data)):
    out = []
    cnima = str()
    end = data.iloc[i]['pred_end_stop_ID']
    start = data.iloc[i]['pred_start_stop_ID']
    start_time = pd.to_datetime(data.iloc[i]['realTime'],format='%Y-%m-%d %H:%M:%S')#data.iloc[i]['realTime']
    last = data.iloc[i]['lastStation_NO']
    if last >= start:
        last = -1
    if last == -1:
        last = start -1
        last_time = start_time# data.iloc[i]['lastStation_Time']
    else:
        last_time = pd.to_datetime(data.iloc[i]['lastStation_Time'],format='%Y-%m-%d %H:%M:%S')#data.iloc[i]['lastStation_Time']
    N = end -last # 7-2

    result = y_pred[count:count+N] # 0 1 2 3 4
    count = count + N
    temp = result[0]
    result[0] = result[0] * 0.525 #- (start_time - last_time).seconds
    M = start - last # 5-2
    if M != 0:
        out.append(result[:M].sum()) # 0 1 2
    else:
        out.append(result[0])

#     if out[0] <= MIN:#MIN
#         out[0] = MIN
        # print(i,data.iloc[i]['LINE'],temp,(start_time - last_time).seconds)
    if out[0] > MAX:
        out[0] = MAX
#     if (start_time - last_time).seconds > MAX:
#         out[0] = temp
    for j in range(1,end-start+1):
        n = j + M - 1 # 3 4
        if result[n] > MAX:
            result[n] = MAX
        if result[n] < MIN:
            result[n] = MIN
        out.append(out[j-1] + result[n])
        # if out[j] >= 3600:
        #     out[j] = 3500
        #     print(i,j, '3500!')
    flag = 0
    if out[-1] >= 3600:
        flag = 1
    for x in range(0,end-start+1):
        if flag == 1:
            out[x] = out[x] * 3500 / out[-1]#3500
        cnima = cnima + str(out[x]) + ";"  # abs
    time_all.append(cnima)
    print(i,cnima)
print(count)
sub["pred_timeStamps"] = time_all
del sub['O_UP']

col_list = []
for i in list(sub.columns):
    col_list.append(i.strip())

sub.columns = col_list
sub.to_csv("./data/answers_0815.csv",index=False)