import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from xgboost import plot_importance
from xgboost import XGBClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
'''
evt_lbl
0. 123-456-789 (evt_123_*)
1. 123-456 (evt_12_*)
2. 123 (evt_1_*)
3. 456 (evt_2_*)
4. 789 (evt_3_*)
'''
def load_data():
    # 读取训练集和测试集的log和flg
    df_train_log = pd.read_csv('../../data/input/train/train_log.csv',sep='\t',parse_dates = ['OCC_TIM'])
    df_test_log = pd.read_csv('../../data/input/test/test_log.csv',sep='\t',parse_dates = ['OCC_TIM'])
    df_train_flg = pd.read_csv('../../data/input/train/train_flg.csv',sep='\t')
    df_test_flg = pd.read_csv('../../data/input/submit_sample.csv',sep='\t')
    df_test_flg['FLAG'] = -1
    del df_test_flg['RST']
    # 切分log
    df_train_log['evt_123'] = df_train_log['EVT_LBL']
    df_train_log['evt_12'] = df_train_log['EVT_LBL'].map(lambda x : str.split(x,sep='-')[0]+'-'+str.split(x,sep='-')[1])
    df_train_log['evt_1'] = df_train_log['EVT_LBL'].map(lambda x : str.split(x,sep='-')[0])
    df_train_log['evt_2'] = df_train_log['EVT_LBL'].map(lambda x : str.split(x,sep='-')[1])
    df_train_log['evt_3'] = df_train_log['EVT_LBL'].map(lambda x : str.split(x,sep='-')[2])
    df_test_log['evt_123'] = df_test_log['EVT_LBL']
    df_test_log['evt_12'] = df_test_log['EVT_LBL'].map(lambda x : str.split(x,sep='-')[0]+'-'+str.split(x,sep='-')[1])
    df_test_log['evt_1'] = df_test_log['EVT_LBL'].map(lambda x : str.split(x,sep='-')[0])
    df_test_log['evt_2'] = df_test_log['EVT_LBL'].map(lambda x : str.split(x,sep='-')[1])
    df_test_log['evt_3'] = df_test_log['EVT_LBL'].map(lambda x : str.split(x,sep='-')[2])
    # 合并训练集和测试集
    df_log = pd.concat([df_train_log, df_test_log],copy=False)
    df_flg = pd.concat([df_train_flg, df_test_flg],copy=False)
    return df_log, df_flg, df_train_log, df_test_log, df_train_flg, df_test_flg

def get_data(df_log, df_flg, name):
    df_sub_log = df_log.groupby(['USRID',name])['USRID'].count().unstack()
    print(df_sub_log.shape)
    df_all = pd.merge(df_sub_log,df_flg,left_index=True,right_on='USRID',how='right')
    df_all.fillna(0,inplace=True)
    print(df_all.shape)

    return df_all

def log_EVTLBL_STA(data,feature_list, name):
    df = data.groupby(['USRID',name])['USRID'].count().unstack()
    df_new = pd.DataFrame()
    for i in feature_list:
        try:
            df_new[i] = df[i]
        except:
            df_new[i] = 0
    df_new.index = df.index
    df_new.fillna(0,inplace=True)
    cols = df_new.columns
    for col in cols:
        df_new.rename(columns={col:name+'_'+col}, inplace = True)
    # df_new['df_new_count0'] = df_new.apply(count_0,axis=1)
    # df_new['df_new_countnone0'] = df_new.apply(count_none0,axis=1)       
    return df_new

def LGB_predict(data, n, name):
    # 获取训练集和测试集
    train = data[data['FLAG']!=-1]
    test = data[data['FLAG']==-1]
    train_x = train.drop(['USRID','FLAG'],axis=1)
    train_y = train['FLAG']
    test_x = test.drop(['USRID','FLAG'],axis=1)
    # 训练
    # clf = XGBClassifier(n_estimators=30,max_depth=5)
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=100, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=30)
    # clf.fit(train_x, train_y)
    # 得到训练结果（概率）作为特征
    train_pre = clf.predict_proba(train_x)[:,1]
    print(train_pre.shape)
    test_pre = clf.predict_proba(test_x)[:,1]
    train_id = train[['USRID']]
    test_id = test[['USRID']]
    train_id[name+'_pre'] = train_pre
    test_id[name+'_pre'] = test_pre
    # 通过模型选取最重要的n个特征
    clf = XGBClassifier(n_estimators=30,max_depth=5)
    clf.fit(train_x, train_y)
    imp = clf.feature_importances_
    #### 打印图形
    impp = imp
    sorted_val = sorted(impp,reverse=True)
    sorted_val = sorted_val[:100]
    plt.bar(range(len(sorted_val)), sorted_val,color='rgb')  
    plt.show()  
    ####
    names = train_x.columns
    d={}
    for i in range(len(names)):
        d[names[i]] = imp[i]
        
    d = sorted(d.items(),key=lambda x:x[1],reverse=True)
    d = d[0:n]
    feture_list=[j[0] for j in d]
    print(feture_list)

    return train_id, test_id, feture_list
'''
evt_lbl
0. 123-456-789 (evt_123_*)  :   618
1. 123-456 (evt_12_*)       :   180
2. 123 (evt_1_*)            :   23
3. 456 (evt_2_*)            :   180
4. 789 (evt_3_*)            :   618
'''
if __name__ == '__main__':
    # 加载数据
    df_log, df_flg, df_train_log, df_test_log, df_train_flg, df_test_flg = load_data()
    evt_list = ['evt_123','evt_12']
    n_list = [50,10]
    df_train_evt_lbl = df_train_flg[['USRID']]
    df_test_evt_lbl = df_test_flg[['USRID']]
    f_list = []
    for n, name in zip(n_list, evt_list):
        data = get_data(df_log, df_flg, name)

        train_pre, test_pre, feature_list = LGB_predict(data, n, name)
        f_list.append(feature_list)
        print('train_pre:', train_pre.shape)
        # feature_list = feture_imp(data,50)
        train_evt_lbl = log_EVTLBL_STA(df_train_log, feature_list,name)
        test_evt_lbl = log_EVTLBL_STA(df_test_log, feature_list,name)
        print('train_evt_lbl', train_evt_lbl.shape)

        df_train_evt_lbl = pd.merge(df_train_evt_lbl,train_pre,on=['USRID'],how='left',copy=False)
        df_train_evt_lbl = pd.merge(df_train_evt_lbl,train_evt_lbl,on=['USRID'],how='left',copy=False)
        df_test_evt_lbl = pd.merge(df_test_evt_lbl,test_pre,on=['USRID'],how='left',copy=False)
        df_test_evt_lbl = pd.merge(df_test_evt_lbl,test_evt_lbl,on=['USRID'],how='left',copy=False)
    print(df_train_evt_lbl.shape)

    df_train_evt_lbl.fillna(0,inplace=True)
    df_test_evt_lbl.fillna(0,inplace=True)

    df_train_evt_lbl.to_csv('../../data/output/feature/train_evt_lbl.csv',index=None)
    df_test_evt_lbl.to_csv('../../data/output/feature/test_evt_lbl.csv',index=None)
