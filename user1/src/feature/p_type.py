import pandas as pd
from functools import reduce

# def count_none0(s):
#     count_none0 = 0
#     for i in s:
#         if i != 0:
#             count_none0 += 1
#     return count_none0

# def count_0(s):
#     count_0 = 0
#     for i in s:
#         if i == 0:
#             count_0 += 1
#     return count_0

def get_type_click(data):
    type_click = []
    for i in range(3):
        type_c = data.groupby(by=['USRID'], as_index=False)['TCH_TYP'].agg({'type_%d'%i: lambda x:reduce(lambda a,b:a+(b==i), x, 0)})
        type_click.append(type_c)
    return type_click

def pross(path):
    df_log = pd.read_csv('../../data/input/'+path+'/'+path+'_log.csv',sep='\t',parse_dates = ['OCC_TIM'])

    df_type = pd.DataFrame()
    for i in get_type_click(df_log):
        try:
            df_type = pd.merge(df_type, i, on=['USRID'], how='left')
        except:
            df_type = i

    df_type.fillna(0,inplace=True)
    return df_type

if __name__ == '__main__':
    df_all_train = pross('train')
    print(df_all_train.shape)
    print('save train...')
    df_all_train.to_csv('../../data/output/feature/train_type.csv', index=None)

    df_all_test = pross('test')
    print(df_all_test.shape)
    print('save test...')
    df_all_test.to_csv('../../data/output/feature/test_type.csv', index=None)
    print('success')
