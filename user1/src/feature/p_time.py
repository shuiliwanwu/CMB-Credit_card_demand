import pandas as pd
import numpy as np
import time
from functools import reduce

def get_in(data):
    first_in = data.groupby(by=['USRID'], as_index=False)['day'].agg({'first_in': min})
    last_in = data.groupby(by=['USRID'], as_index=False)['day'].agg({'last_in': max})
    return first_in, last_in

def get_weekday_click(data):
    weekday = []
    for i in range(7):
        wd = data.groupby(by=['USRID'], as_index=False)['weekday'].agg({'wd_%d'%(i+1): lambda x:reduce(lambda a,b:a+(b==i+1), x, 0)})
        weekday.append(wd)
    return weekday

def get_day_click(data):
    day_click = []
    for i in range(1,32):
        lwc = data.groupby(by=['USRID'], as_index=False)['day'].agg({'lwc_%d'%i: lambda x:reduce(lambda a,b:a+(b==i), x, 0)})
        day_click.append(lwc)
    return day_click

def get_hour_click(data):
    hour_click = []
    for i in range(1,25):
        hrc = data.groupby(by=['USRID'], as_index=False)['hour'].agg({'hrc_%d'%i: lambda x:reduce(lambda a,b:a+(b==i), x, 0)})
        hour_click.append(hrc)
    return hour_click

def get_next_time(data):
    data_sort = data
    data_sort = data_sort.sort_values(['USRID','OCC_TIM'])
    data_sort['next_time'] = data_sort.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(lambda x:x.seconds).apply(np.abs)

    next_time = data_sort.groupby(by=['USRID'],as_index=False)['next_time'].agg({
        'next_time_mean':lambda x : np.mean(x),
        'next_time_std':lambda x : np.std(x),
        'next_time_min':lambda x : np.min(x),
        'next_time_max':lambda x : np.max(x)})
    return next_time

def pross(path):
    df_log = pd.read_csv('../../data/input/'+path+'/'+path+'_log.csv',sep='\t',parse_dates = ['OCC_TIM'])
    df_log['day'] = df_log['OCC_TIM'].map(lambda x:x.day)
    df_log['hour'] = df_log['OCC_TIM'].map(lambda x:x.hour)
    df_log['weekday'] = df_log['OCC_TIM'].map(lambda x:x.isoweekday())
    first_in, last_in = get_in(df_log)
    df_time = first_in
    df_time = pd.merge(df_time, last_in, on=['USRID'], how='left')
    for i in get_hour_click(df_log):
        df_time = pd.merge(df_time, i, on=['USRID'], how='left')
    for i in get_day_click(df_log):
        df_time = pd.merge(df_time, i,on=['USRID'],how='left')
    for i in get_weekday_click(df_log):
        df_time = pd.merge(df_time, i, on=['USRID'], how='left')
    df_time = pd.merge(df_time, get_next_time(df_log), on=['USRID'], how='left')
    df_time.fillna(0,inplace=True)
    print(df_time)
    
    print('saved')
    return df_time

if __name__ == '__main__':
    df_time_train = pross('train')
    df_time_train.to_csv('../../data/output/feature/train_time.csv', index=None)
    
    df_time_test = pross('test')
    df_time_test.to_csv('../../data/output/feature/test_time.csv', index=None)
