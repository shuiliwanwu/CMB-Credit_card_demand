
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:



colname2 = ['uid', 'module', 'time', 'type']

train_log = pd.read_csv('../data/input/train/train_log.csv', sep='\t', skiprows=1, names=colname2)
test_log = pd.read_csv('../data/input/test/test_log.csv', sep='\t', skiprows=1, names=colname2)
log_set = pd.concat([train_log, test_log])


# In[3]:


week1=log_set[(log_set.time>='2018-03-01 00:00:00') & (log_set.time<'2018-03-08 00:00:00')]

week2=log_set[(log_set.time>='2018-03-08 00:00:00') & (log_set.time<'2018-03-15 00:00:00')]

week3=log_set[(log_set.time>='2018-03-15 00:00:00') & (log_set.time<'2018-03-22 00:00:00')]

week4=log_set[(log_set.time>='2018-03-22 00:00:00') & (log_set.time<'2018-03-29 00:00:00')]

weeklast=log_set[(log_set.time>='2018-03-29 00:00:00')]


# In[4]:


week1b=week1[week1.time>='2018-03-07 00:00:00']

week1bEVT_LBL_len = week1b.groupby(by= ['uid'], as_index = False)['module'].agg({'week1bEVT_LBL_len':len})
week1bEVT_LBL_set_len = week1b.groupby(by= ['uid'], as_index = False)['module'].agg({'week1bEVT_LBL_set_len':lambda x:len(set(x))})
week1btype_len = week1b.groupby(by= ['uid'], as_index = False)['type'].agg({'week1btype_len':len})
week1btype_set_len = week1b.groupby(by= ['uid'], as_index = False)['type'].agg({'week1btype_set_len':lambda x:len(set(x))})

week1a=week1[week1.time<'2018-03-07 00:00:00']

week1aEVT_LaL_len = week1a.groupby(by= ['uid'], as_index = False)['module'].agg({'week1aEVT_LaL_len':len})
week1aEVT_LaL_set_len = week1a.groupby(by= ['uid'], as_index = False)['module'].agg({'week1aEVT_LaL_set_len':lambda x:len(set(x))})


# In[5]:


week2b=week2[week2.time>='2018-03-14 00:00:00']

week2bEVT_LBL_len = week2b.groupby(by= ['uid'], as_index = False)['module'].agg({'week2bEVT_LBL_len':len})
week2bEVT_LBL_set_len = week2b.groupby(by= ['uid'], as_index = False)['module'].agg({'week2bEVT_LBL_set_len':lambda x:len(set(x))})

week2btype_len = week2b.groupby(by= ['uid'], as_index = False)['type'].agg({'week2btype_len':len})
week2btype_set_len = week2b.groupby(by= ['uid'], as_index = False)['type'].agg({'week2btype_set_len':lambda x:len(set(x))})


week2a=week2[week2.time<'2018-03-14 00:00:00']

week2aEVT_LaL_len = week2a.groupby(by= ['uid'], as_index = False)['module'].agg({'week2aEVT_LaL_len':len})
week2aEVT_LaL_set_len = week2a.groupby(by= ['uid'], as_index = False)['module'].agg({'week2aEVT_LaL_set_len':lambda x:len(set(x))})


# In[6]:


week3b=week3[week3.time>='2018-03-21 00:00:00']

week3bEVT_LBL_len = week3b.groupby(by= ['uid'], as_index = False)['module'].agg({'week3bEVT_LBL_len':len})
week3bEVT_LBL_set_len = week3b.groupby(by= ['uid'], as_index = False)['module'].agg({'week3bEVT_LBL_set_len':lambda x:len(set(x))})

week3btype_len = week3b.groupby(by= ['uid'], as_index = False)['type'].agg({'week3btype_len':len})
week3btype_set_len = week3b.groupby(by= ['uid'], as_index = False)['type'].agg({'week3btype_set_len':lambda x:len(set(x))})

week3a=week3[week3.time<'2018-03-21 00:00:00']

week3aEVT_LaL_len = week3a.groupby(by= ['uid'], as_index = False)['module'].agg({'week3aEVT_LaL_len':len})
week3aEVT_LaL_set_len = week3a.groupby(by= ['uid'], as_index = False)['module'].agg({'week3aEVT_LaL_set_len':lambda x:len(set(x))})


# In[7]:


week4b=week4[week4.time>='2018-03-28 00:00:00']

week4bEVT_LBL_len = week4b.groupby(by= ['uid'], as_index = False)['module'].agg({'week4bEVT_LBL_len':len})
week4bEVT_LBL_set_len = week4b.groupby(by= ['uid'], as_index = False)['module'].agg({'week4bEVT_LBL_set_len':lambda x:len(set(x))})

week4btype_len = week4b.groupby(by= ['uid'], as_index = False)['type'].agg({'week4btype_len':len})
week4btype_set_len = week4b.groupby(by= ['uid'], as_index = False)['type'].agg({'week4btype_set_len':lambda x:len(set(x))})

week4a=week4[week4.time<'2018-03-28 00:00:00']

week4aEVT_LaL_len = week4a.groupby(by= ['uid'], as_index = False)['module'].agg({'week4aEVT_LaL_len':len})
week4aEVT_LaL_set_len = week4a.groupby(by= ['uid'], as_index = False)['module'].agg({'week4aEVT_LaL_set_len':lambda x:len(set(x))})


# In[8]:


weeklastEVT_LaL_len = weeklast.groupby(by= ['uid'], as_index = False)['module'].agg({'weeklastEVT_LaL_len':len})
weeklastEVT_LaL_set_len = weeklast.groupby(by= ['uid'], as_index = False)['module'].agg({'weeklastEVT_LaL_set_len':lambda x:len(set(x))})
weeklasttype_len = weeklast.groupby(by= ['uid'], as_index = False)['type'].agg({'weeklasttype_len':len})
weeklasttype_set_len = weeklast.groupby(by= ['uid'], as_index = False)['type'].agg({'weeklasttype_set_len':lambda x:len(set(x))})


# In[9]:


weekfeature=pd.DataFrame()


# In[10]:


zz=log_set.groupby('uid')['type'].agg('mean').reset_index()
weekfeature['uid']=zz['uid']


# In[11]:


weekfeature=pd.merge(weekfeature,week1aEVT_LaL_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week1aEVT_LaL_set_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week1bEVT_LBL_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week1bEVT_LBL_set_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week1btype_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week1btype_set_len,on='uid',how='left')


# In[12]:


weekfeature=pd.merge(weekfeature,week2aEVT_LaL_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week2aEVT_LaL_set_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week2bEVT_LBL_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week2bEVT_LBL_set_len,on='uid',how='left')

weekfeature=pd.merge(weekfeature,week2btype_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week2btype_set_len,on='uid',how='left')


# In[13]:


weekfeature=pd.merge(weekfeature,week3aEVT_LaL_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week3aEVT_LaL_set_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week3bEVT_LBL_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week3bEVT_LBL_set_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week3btype_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week3btype_set_len,on='uid',how='left')


# In[14]:


weekfeature=pd.merge(weekfeature,week4aEVT_LaL_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week4aEVT_LaL_set_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week4bEVT_LBL_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week4bEVT_LBL_set_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week4btype_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,week4btype_set_len,on='uid',how='left')


# In[15]:


weekfeature=pd.merge(weekfeature,weeklastEVT_LaL_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,weeklastEVT_LaL_set_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,weeklasttype_len,on='uid',how='left')
weekfeature=pd.merge(weekfeature,weeklasttype_set_len,on='uid',how='left')


# In[16]:


weekfeature=weekfeature.fillna(0)


# In[17]:


weekfeature['diffweek21bEVT_LBL_len']=weekfeature['week2bEVT_LBL_len']-weekfeature['week1bEVT_LBL_len']
weekfeature['diffweek21bEVT_LBL_set_len']=weekfeature['week2bEVT_LBL_set_len']-weekfeature['week1bEVT_LBL_set_len']
weekfeature['diffweek32bEVT_LBL_len']=weekfeature['week3bEVT_LBL_len']-weekfeature['week2bEVT_LBL_len']
weekfeature['diffweek32bEVT_LBL_set_len']=weekfeature['week3bEVT_LBL_set_len']-weekfeature['week2bEVT_LBL_set_len']

weekfeature['diffweek43bEVT_LBL_len']=weekfeature['week4bEVT_LBL_len']-weekfeature['week3bEVT_LBL_len']
weekfeature['diffweek43bEVT_LBL_set_len']=weekfeature['week4bEVT_LBL_set_len']-weekfeature['week3bEVT_LBL_set_len']


# In[18]:


weekfeature['USRID']=weekfeature['uid']


# In[19]:


weekfeature.pop('uid')


# In[20]:


weekfeature.info()


# In[21]:


weekfeature[['USRID','diffweek21bEVT_LBL_set_len','diffweek32bEVT_LBL_len','diffweek32bEVT_LBL_set_len','diffweek43bEVT_LBL_len','diffweek43bEVT_LBL_set_len']].to_csv('week3.csv',index=None)

