
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

colname1 = ['v'+str(i) for i in range(1,31)] + ['uid']

train_agg = pd.read_csv('../data/input/train/train_agg.csv', sep='\t', skiprows=1, names=colname1)
test_agg = pd.read_csv('../data/input/test/test_agg.csv', sep='\t', skiprows=1, names=colname1)

colname2 = ['uid', 'module', 'time', 'type']
train_log = pd.read_csv('../data/input/train/train_log.csv', sep='\t', skiprows=1, names=colname2)
test_log = pd.read_csv('../data/input/test/test_log.csv', sep='\t', skiprows=1, names=colname2)

log_set = pd.concat([train_log, test_log])
log_set['time2'] = log_set.time.apply(lambda x: pd.to_datetime(x))



import pandas as pd
test_set=pd.read_csv('data/test.csv')

zz=pd.DataFrame()
zz['colu']=test_set.columns
zz.to_csv('columns.csv')

import numpy as np
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
import sys
import scipy as sp
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import time
import gc
log_set['date']=log_set['time'].apply(lambda x : time.strptime(x, "%Y-%m-%d %H:%M:%S"))
log_set['timediff']=log_set.date.apply(lambda x : int(time.mktime(x)))
log_set['timediff']=log_set['timediff']-1519833600

#%%
log_set['day'] = log_set.time2.apply(lambda x: x.day)
log_set['dayofweek']  = log_set.time2.apply(lambda x: x.isoweekday())
log_set['isWen'] = log_set.dayofweek.apply(lambda x: 1 if x==3 else 0)
log_set['isweekend'] = log_set.dayofweek.apply(lambda x: 1 if x in [6, 7] else 0)
log_set['exactTime'] = log_set.time.apply(lambda x: x[-8:])

log_set['promotiontime1'] = log_set.exactTime.apply(lambda x: 1 if '09:30:00'<=x<='10:30:00' else 0)
log_set['promotiontime2'] = log_set.exactTime.apply(lambda x: 1 if '14:50:00'<=x<='15:10:00' else 0)
module_count = dict(log_set.module.value_counts())
log_set['module'] = log_set.module.apply(lambda x: 'other' if module_count[x]<=100 else x)
#%%
sample = []
module_count_sort = log_set.module.value_counts().argsort()
for g in log_set.groupby('uid'):
    c_log = [0] * len(module_count_sort)
    userdf = g[1]
    l = []
    l.append(g[0])
    l.append(userdf.shape[0]) # l0登录次数
    l.append(len(np.unique(userdf.day))) # l1 登录天数
    l.append(31 - userdf.day.max()) # l2 最后登录离31号
    l.append(userdf.isWen.sum()) # l3 周三登录次数
    l.append(userdf.promotiontime2.sum()) # l4 活动时间1登录
    promotiontime1 = userdf[(userdf.isWen==1) & (userdf.promotiontime1==1)]
    l.append(promotiontime1.shape[0]) # l5 活动时间2登录
    #l.append(userdf.day.mode()) # l6 经常登录的天数,
    l.append(userdf.isweekend.sum()) # l7 周末登录
    l.append(userdf.shape[0] - userdf.isweekend.sum()) #l8 工作日登录
    for _,v in enumerate(userdf.module.values):
        c_log[module_count_sort[v]] +=1
    l = l + c_log
    sample.append(l)

log_name = ['USRID'] + ['l'+ str(i) for i in range(len(l)-1)]
log_feature = pd.DataFrame(sample, columns=log_name)



log_feature.to_csv('log_feature.csv',index=None)

