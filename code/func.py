def get_last_station(bus1,target):
    # bus1[(bus1['O_SPEED'] == 0) & (bus1['O_LONGITUDE'] != 0.0) & (bus1['O_LATITUDE'] != 0.0)]
    len = bus1.shape[0]
    for i in range(len-1):
        station = bus1['O_NEXTSTATIONNO'].iloc[i]
        on = bus1['O_UP'].iloc[i]

        next_time = bus1['O_TIME'].iloc[i+1]
        next_station = bus1['O_NEXTSTATIONNO'].iloc[i+1]
        if (station == target and next_station == (target+1)):
            return bus1.iloc[i+1]

df = pd.DataFrame()
count = 0
for row in toBePredict.itertuples():
    import datetime
    line = row[2] # 线路
    carNo = row[3] # 车编号
    currtime = row[7] # 此时时间
    target = row[5] - 1
    twominute = datetime.timedelta(minutes=-60) + currtime
    
    bus1 = assigment[(assigment['O_LINENO']==line) & (assigment['O_TERMINALNO']==carNo)]
    bus1 = bus1[(bus1['O_TIME']<currtime) & (bus1['O_TIME']>twominute)]
    last_station = get_last_station(bus1,target)
    while last_station is None:
        target -= 1
        last_station = get_last_station(bus1,target)
    #####
    
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
    lastStation_NO = target
    lastStation_Time = last_station[2]
    LONGITUDE = last_station[3]
    LATITUDE = last_station[4]
    UP = last_station[9]
    
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
    
    if row[0] % 100 == 0:
        print('finishded {}\n'.format(row[0]))