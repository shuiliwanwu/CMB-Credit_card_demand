import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import sklearn.svm as svm
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def load_data():
    # 读取个人信息 agg
    train_agg = pd.read_csv('../data/input/train/train_agg.csv',sep='\t')
    test_agg = pd.read_csv('../data/input/test/test_agg.csv',sep='\t')
    agg = pd.concat([train_agg,test_agg],copy=False)

    # 读取flg
    train_flg = pd.read_csv('../data/input/train/train_flg.csv',sep='\t')
    test_flg = pd.read_csv('../data/input/submit_sample.csv',sep='\t')
    test_flg['FLAG'] = -1
    del test_flg['RST']
    flg = pd.concat([train_flg,test_flg],copy=False)

    data = pd.merge(agg,flg,on=['USRID'],how='left',copy=False)

    # 读取时间特征
    train_time = pd.read_csv('../data/output/feature/train_time.csv')
    test_time = pd.read_csv('../data/output/feature/test_time.csv')
    all_time = pd.concat([train_time, test_time], copy=False)

    data = pd.merge(data, all_time, on=['USRID'], how='left', copy=False)

    # 读取统计特征 count
    train_count = pd.read_csv('../data/output/feature/train_count.csv')
    test_count = pd.read_csv('../data/output/feature/test_count.csv')
    count = pd.concat([train_count, test_count], copy=False)

    data = pd.merge(data, count, on=['USRID'], how='left', copy=False)

    # 读取模块特征
    train_evt_lbl = pd.read_csv('../data/output/feature/train_evt_lbl.csv')
    test_evt_lbl = pd.read_csv('../data/output/feature/test_evt_lbl.csv')
    evt_lbl = pd.concat([train_evt_lbl, test_evt_lbl], copy=False)

    data = pd.merge(data, evt_lbl, on=['USRID'], how='left', copy=False)

    # 读取登录类型特征
    train_type = pd.read_csv('../data/output/feature/train_type.csv')
    test_type = pd.read_csv('../data/output/feature/test_type.csv')
    all_type = pd.concat([train_type, test_type], copy=False)

    data = pd.merge(data, all_type, on=['USRID'], how='left', copy=False)
    data.fillna(0,inplace=True)
    data.to_csv('../data/output/feature/train_all.csv', index=None)
    return data

def xgb_predict(X_train,X_test,y_train,y_test, test):
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]

    params = {
        ## 通用参数
        'booster': 'gbtree',
        'silent':1,  # 当这个参数值为1时，静默模式开启，不会输出任何信息。
        ## booster参数
        'eta': 0.02,
        'min_child_weight': 9,  # 2 3
        'max_depth': 4, #7
        #'gamma'
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        ## 学习目标参数
        'objective':'binary:logistic',
        'eval_metric': 'auc',
        # 'seed':2333
    }

    plst = list(params.items())
    num_rounds = 5000 # 迭代次数
    
    print('Start training...')
    # 训练模型
    model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100,verbose_eval=100)

    # print('Save lgb model...')
    # save model to file
    # gbm.save_model('model.txt')

    xgb_x_test = xgb.DMatrix(test)
    print('Start predicting...')
    y_pred = model.predict(xgb_test)
    cv = roc_auc_score(y_test,y_pred)
    pre = model.predict(xgb_x_test)
    return cv, pre

def lgb_predict(X_train,X_test,y_train,y_test, test):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 20,
        'max_depth': 6,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        # 'is_unbalance': 'true' # 样本分布非平衡数据集
    }

    print('Start training...')
    # 训练模型
    lgb_model = lgb.train(params,lgb_train,num_boost_round=2000,valid_sets=[lgb_train, lgb_eval],verbose_eval=100,early_stopping_rounds=100)

    # print('Save lgb model...')
    # save model to file
    # gbm.save_model('model.txt')

    print('Start predicting...')
    y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    cv = roc_auc_score(y_test,y_pred)
    pre = lgb_model.predict(test, num_iteration=lgb_model.best_iteration)
    return cv, pre

def solve(data):
    # 提取测试集和训练集合
    train = data[data['FLAG']!=-1]
    test = data[data['FLAG']==-1]
    print('train',train.shape)
    print('test',test.shape)

    # 构造数据
    # 提取userid和单独把标签赋值一个变量
    train_userid = train.pop('USRID')
    y = train.pop('FLAG')
    noise = []
    train = train.drop(noise, axis=1)
    col = train.columns
    print('train column:', col)
    X = train[col].values

    test_userid = test.pop('USRID')
    test_y = test.pop('FLAG')
    test = test[col].values

    # 5折交叉验证
    N = 5
    skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=3)

    cv_list = []
    pre_list = []

    for train_in,test_in in skf.split(X,y):
        X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

        cv, pre = xgb_predict(X_train,X_test,y_train,y_test,test)

        cv_list.append(cv)
        pre_list.append(pre)

    # 取平均值作为预测结果
    avg_pre = 0
    for i in pre_list:
        avg_pre = avg_pre + i
    avg_pre = avg_pre / N

    # 打印线下分数
    print('cv:',np.mean(cv_list))

    # 保存 result
    res = pd.DataFrame()
    res['USRID'] = list(test_userid.values)
    res['RST'] = list(avg_pre)

    import time
    time_date = time.strftime('%m-%d_%H:%M:',time.localtime(time.time()))
    #res.to_csv('../data/output/result/%s_%s.csv'%(str(time_date),str(np.mean(cv_list).__format__('.6f')).split('.')[1]),index=False,sep='\t')
	res.to_csv('../../result/user1_xgb.csv',index=False,sep='\t')

if __name__ == '__main__':
    data = load_data()
    print(data.shape)
    solve(data)