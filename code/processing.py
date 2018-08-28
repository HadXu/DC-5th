
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime as dt


from contextlib import contextmanager

@contextmanager
def timer(name):
    start_time = time.time()
    yield
    print(f'[{name} done in {time.time() - start_time:.2f} s]')

def get_last_station(bus1):
    # bus1[(bus1['O_SPEED'] == 0) & (bus1['O_LONGITUDE'] != 0.0) & (bus1['O_LATITUDE'] != 0.0)]
    len = bus1.shape[0]
    for i in range(len-1,0,-1):
        station = bus1['O_NEXTSTATIONNO'].iloc[i-1]

        next_station = bus1['O_NEXTSTATIONNO'].iloc[i]

        if (next_station > station):
            return next_station-1,bus1.iloc[i]['O_UP'],bus1.iloc[i]
        elif (next_station < station):
            return next_station-1,bus1.iloc[i]['O_UP'],bus1.iloc[i-1]

# read

assigment = pd.DataFrame()
with timer('read assigment csv'):
    for day in ['25','26','27','28','29','30','31']:
        temp = pd.read_csv("data/train201710"+day+".csv", sep=",")
        temp['O_TIME'] = pd.to_datetime('2017-10-'+day+' '+temp['O_TIME'],format='%Y-%m-%d %H:%M:%S')
        temp = temp.groupby(['O_LINENO', 'O_TERMINALNO']).apply(lambda x:x.sort_values(by='O_TIME', ascending=True))
        assigment = pd.concat([assigment,temp])
assigment['O_DATA'] = assigment['O_TIME'].dt.day

with timer('read pred'):
    toBePredict = pd.read_csv('data/toBePredicted.csv',index_col=False)
toBePredict['realTime'] = pd.to_datetime('2017-'+toBePredict['O_DATA']+' '+toBePredict['predHour'],format='%Y-%m-%d %H:%M:%S')



df = pd.DataFrame()
count = 0

for row in toBePredict.itertuples():
    import datetime
    line = row[2] # 线路
    carNo = row[3] # 车编号
    currtime = row[7] # 此时时间
    twominute = datetime.timedelta(minutes=-60) + currtime
    
    bus1 = assigment[(assigment['O_TERMINALNO']==carNo) & (assigment['O_LINENO']==line)]
    bus1 = bus1[(bus1['O_TIME']<currtime) & (bus1['O_TIME']>twominute)]
    
    if get_last_station(bus1) is None:
        #如果发生下面都不动了
        try:
            res = bus1.iloc[-1]
            lastStation_NO = res[11]-1
            lastStation_Time = res[2]
            LONGITUDE = res[3]
            LATITUDE = res[4]
            UP = res[9]
            with open('problems.txt','a+') as f:
                f.write(str(tuple(row))+'\n')
                print(str(tuple(row))+'\n')
        except:
            res = -1
            lastStation_NO = -1
            lastStation_Time = -1
            LONGITUDE = -1
            LATITUDE = -1
            UP = -1
            with open('problems.txt','a+') as f:
                f.write(str(tuple(row))+'\n')
                print(str(tuple(row))+'\n')
    else:
        last_station_NO, up, last_station_info= get_last_station(bus1)
        # 如果
        lastStation_NO = last_station_NO
        lastStation_Time = last_station_info[2]
        LONGITUDE = last_station_info[3]
        LATITUDE = last_station_info[4]
        UP = up
    
    
    """
    O_DATA O_LINENO O_TERMINALNO predHour pred_start_stop_ID pred_end_stop_ID 
    
    realTime lastStation lastStation_Time LONGITUDE LATITUDE UP
    
    """
    DATA = row[1]
    LINE = row[2]
    TERMINALNO = row[3]
    predHour = row[4]
    pred_start_stop_ID = row[5]
    pred_end_stop_ID = row[6]
    realTime = row[7]
    
#     lastStation_NO = last_station_NO
#     lastStation_Time = last_station_info[2]
#     LONGITUDE = last_station_info[3]
#     LATITUDE = last_station_info[4]
#     UP = last_station_info[9]
    
    
    temp = pd.DataFrame({
        'DATA':[DATA],
        'LINE':[LINE],
        'TERMINALNO':[TERMINALNO],
        'predHour':[predHour],
        'pred_start_stop_ID':[pred_start_stop_ID],
        'pred_end_stop_ID':[pred_end_stop_ID],
        'realTime':[realTime],
        'lastStation_NO':[lastStation_NO],
        'lastStation_Time':[lastStation_Time],
        'LONGITUDE':[LONGITUDE],
        'LATITUDE':[LATITUDE],
        'UP':[UP],
    })
    df = pd.concat([df,temp])
    print('finishded {}\n'.format(row[0]))
df.to_csv('toBePredicted_525.csv',index=False)