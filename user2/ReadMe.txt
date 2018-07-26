xgboost版本：0.72  python 3.6

train，test文件夹 为比赛提供的数据

code： 个人代码部分。

特征部分代码： evt_one.ipynb:生成final_feature.csv
		CMB 0704.ipynb：生成lbl_one_shift.csv
		evt数量模块特征.ipynb :evt_level_click_count.csv，evt_level_one_click_count.csv，evt_level_two_click_count.csv，evt_level_three_click_count.csv，evt_lbl_click_count.csv
		OCC_TIM模块特征.ipynb ：day_click_count.csv，day_statical.csv，hour_statical.csv，minute_statical.csv，weekday_click_count.csv，day_dist_promote.csv
模型部分：综合特征,产生7-10-levelxgb_select_all-test2，7-10-nopromotetimexgb_select_all-test，7-10-xgb_select_all-test

融合：产生     best.csv