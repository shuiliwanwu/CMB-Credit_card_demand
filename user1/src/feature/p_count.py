import pandas as pd
from functools import reduce

def get_count(data):
    click_cnt = data.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'click_cnt':len})
    evt_lbl_cnt = data.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'evt_lbl_cnt':lambda x:len(set(x))})
    day_cnt = data.groupby(by=['USRID'], as_index = False)['day'].agg({'day_cnt': lambda x:len(set(x))})
    week_cnt = data.groupby(by=['USRID'], as_index = False)['weekday'].agg({'week_cnt': lambda x:len(set(x))})
    return click_cnt, evt_lbl_cnt,day_cnt,week_cnt

def pross(path):
    df_log = pd.read_csv('../../data/input/'+path+'/'+path+'_log.csv',sep='\t',parse_dates = ['OCC_TIM'])
    df_log['day'] = df_log['OCC_TIM'].map(lambda x:x.day)
    df_log['weekday'] = df_log['OCC_TIM'].map(lambda x:x.isoweekday())
    click_cnt,evt_lbl_cnt,day_cnt, week_cnt = get_count(df_log)

    df_all = click_cnt
    df_all = pd.merge(df_all, evt_lbl_cnt,on=['USRID'],how='left')
    df_all = pd.merge(df_all,day_cnt,on=['USRID'],how='left')
    df_all = pd.merge(df_all,week_cnt,on=['USRID'],how='left')

    df_all.fillna(0,inplace=True)
    print(df_all)
    
    print('saved')
    return df_all

if __name__ == '__main__':
    df_all_train = pross('train')
    df_all_train.to_csv('../../data/output/feature/train_count.csv', index=None)

    df_all_test = pross('test')
    df_all_test.to_csv('../../data/output/feature/test_count.csv', index=None)
