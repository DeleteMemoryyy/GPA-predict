# -*- coding: UTF-8 -*-
#%% init
import time
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing as prep
from sklearn import ensemble
from sklearn import linear_model as lm
from sklearn import svm as svm
from sklearn import metrics as metrics
from matplotlib import pyplot as plt
import seaborn as sns

# proc_data = pd.ExcelFile('data/ALLDATA.xlsx').parse('Sheet1')
all_data = pd.read_csv('data/ALLDATA.csv')
dpart_rank = pd.read_csv('data/depart_rank.csv')
all_data = pd.concat([all_data, dpart_rank], axis=1)
proc_data = all_data

drop_gpa = 0.5

svr_C = 200
svr_gamma = 0.001
regr_alpha = 11.0
lsr_alpha = 0.0005547
enr_alpha = 0.0009649
enr_l1r = 0.5

rand_seed = 33
fill_in_gpa = 2.35815726

ori_one_hot_columns = ['province', 'gender', 'test_year', 'nation', 'politics', 'color_blind',
                       'stu_type', 'lan_type', 'sub_type', 'birth_year', 'department', 'reward_type']

ori_numerical_columns = ['left_sight', 'right_sight', 'height', 'weight', 'grade',
                         'admit_grade', 'admit_rank', 'center_grade', 'dpart_rank', 'reward_score', 'school_num', 'school_admit_rank',
                         'high_rank', 'rank_var', 'progress', 'patent', 'social', 'prize','competition']

drop_columns = ['grade', 'admit_grade', 'high_school', 'high_rank', 'rank_var', 'progress',
                'color_blind', 'lan_type', 'left_sight', 'right_sight', 'patent']

one_hot_columns = ['province', 'gender', 'birth_year', 'nation', 'politics',                   'test_year', 'stu_type', 'sub_type', 'department', 'reward_type']

numerical_columns = ['admit_rank', 'school_num', 'center_grade','social',
                     'school_admit_rank', 'dpart_rank', 'reward_score', 'competition', 'height', 'weight']

other_columns = ['student_ID', 'GPA', 'test_tag', 'test_ID']

# preprocess features
# drop outlier
for i in range(proc_data.shape[0]):
    if(proc_data['test_tag'][i] != 'test' and proc_data['GPA'][i] <= drop_gpa):
        proc_data = proc_data.drop(i, axis=0)
proc_data.index
proc_data.index = range(proc_data.shape[0])

# fill nan
proc_data['rank_var'] = proc_data['rank_var'].fillna(proc_data['rank_var'].mean())
proc_data['high_rank'] = proc_data['high_rank'].fillna(
    proc_data['high_rank'].mean())
proc_data['progress'] = proc_data['progress'].fillna(proc_data['progress'].mean())
proc_data['admit_grade'] = proc_data['admit_grade'].fillna(
    proc_data['admit_grade'].mean())
proc_data['center_grade'] = proc_data['center_grade'].fillna(0)
proc_data['reward_score'] = proc_data['reward_score'].fillna(0)
proc_data['prize'] = proc_data['prize'].fillna(0)
proc_data['competition'] = proc_data['competition'].fillna(0)
proc_data['patent'] = proc_data['patent'].fillna(0)
proc_data['social'] = proc_data['social'].fillna(0)

# process birth_year
def process_birth_year(x):
    test_age = x['test_year'] - x['birth_year']
    if (test_age <= 15):
        test_age = 15
    elif (test_age >= 20):
        test_age = 20
    return test_age

    tmp_high_rank = all_data['high_rank']
    for i in range(all_data.shape[0]):
        if(all_data['high_rank'][i] >= 0.5):
            tmp_high_rank[i] = all_data['high_rank'][i] / 350.0
    all_data['high_rank'] = tmp_high_rank

proc_data['birth_year'] = proc_data.apply(process_birth_year, axis=1)

# process sight
def process_sight(x):
    if (x == np.nan):
        return x
    if(x >= 3.0):
        return x
    elif (x >= 2.0 and x < 3.0):
            return 5.3
    elif (x >= 1.5 and x < 2.0):
        return 5.2
    elif (x >= 1.2 and x < 1.5):
        return 5.1
    elif (x >= 1.0 and x < 1.2):
        return 5.0
    elif (x >= 1.0 and x < 1.2):
        return 5.0
    elif (x >= 0.8 and x < 1.0):
        return 4.9
    elif (x >= 0.6 and x < 0.8):
        return 4.8
    elif (x >= 0.5 and x < 0.6):
        return 4.7
    elif (x >= 0.4 and x < 0.5):
        return 4.6
    elif (x >= 0.3 and x < 0.4):
        return 4.5
    elif (x >= 0.25 and x < 0.3):
        return 4.4
    elif (x >= 0.2 and x < 0.25):
        return 4.3
    elif (x >= 0.15 and x < 0.2):
        return 4.2
    elif (x >= 0.12 and x < 0.15):
        return 4.1
    elif (x >= 0.1 and x < 0.12):
        return 4.0
    elif (x >= 0.08 and x < 0.1):
        return 3.9
    elif (x >= 0.08 and x < 0.1):
        return 3.9
    elif (x >= 0.06 and x < 0.08):
        return 3.8
    elif (x >= 0.05 and x < 0.06):
        return 3.7
    elif (x >= 0.04 and x < 0.05):
        return 3.6
    elif (x >= 0.03 and x < 0.04):
        return 3.5
    else:
        return np.nan

proc_data['left_sight'] = proc_data['left_sight'].apply(process_sight)
proc_data['right_sight'] = proc_data['right_sight'].apply(process_sight)
proc_data['left_sight'] = proc_data['left_sight'].fillna(
    proc_data['left_sight'].mean())
proc_data['right_sight'] = proc_data['right_sight'].fillna(
    proc_data['right_sight'].mean())

# process nation
d_nation = {}
temp_nation = []
for i in range(proc_data.shape[0]):
    if (proc_data['nation'][i] in d_nation.keys()):
            d_nation[proc_data['nation'][i]] += 1
    else:
        d_nation[proc_data['nation'][i]] = 1
for i in range(proc_data.shape[0]):
    if (d_nation[proc_data['nation'][i]] <= 6):
        temp_nation.append('少数民族')
    else:
        temp_nation.append(proc_data['nation'][i])
proc_data['nation'] = temp_nation

# one-hot processing
proc_data[one_hot_columns] = proc_data[one_hot_columns].fillna('Empty')
proc_data = pd.get_dummies(proc_data, columns=one_hot_columns)

# drop features
proc_data = proc_data.drop(drop_columns,axis=1)

# standardization
ss_x = prep.StandardScaler()
ss_y = prep.StandardScaler()
proc_data[numerical_columns] = ss_x.fit_transform(proc_data[numerical_columns].values)

# spilt training data
x_all_train = proc_data[proc_data['test_tag']!='test'].drop(other_columns,axis=1)
x_test = proc_data[proc_data['test_tag'] == 'test'].drop(other_columns, axis=1)
result_data = proc_data['GPA'][proc_data['test_tag']!='test']
y_all_train = ss_y.fit_transform(result_data.values.reshape(-1,1)).ravel()
# x_train,x_valid,y_train,y_valid = train_test_split(x_all_train.values,y_all_train.values,
#                                                 random_state=rand_seed)

#%% regression
# SVR
svr = svm.SVR(C=svr_C, gamma=svr_gamma)
svr_score = -cross_val_score(svr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
svr.fit(x_all_train, y_all_train)
svr_y_all_predict = ss_y.inverse_transform(svr.predict(x_all_train))
svr_y_test_predict = ss_y.inverse_transform(svr.predict(x_test))
print("svr_valid_mse: {}".format(svr_score.mean()))
print("svr_all_mse: {}".format(metrics.mean_squared_error(result_data,svr_y_all_predict)))

# Ridge regression
regr = lm.Ridge(alpha=regr_alpha)
regr_score = -cross_val_score(regr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
regr.fit(x_all_train, y_all_train)
regr_y_all_predict = ss_y.inverse_transform(regr.predict(x_all_train))
regr_y_test_predict = ss_y.inverse_transform(regr.predict(x_test))
print("regr_valid_mse: {}".format(regr_score.mean()))
print("regr_all_mse: {}".format(
    metrics.mean_squared_error(result_data, regr_y_all_predict)))

# Lasso regression
lsr = lm.Lasso(alpha=lsr_alpha)
lsr_score = -cross_val_score(lsr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
lsr.fit(x_all_train, y_all_train)
lsr_y_all_predict = ss_y.inverse_transform(lsr.predict(x_all_train))
lsr_y_test_predict = ss_y.inverse_transform(lsr.predict(x_test))
print("lsr_valid_mse: {}".format(lsr_score.mean()))
print("lsr_all_mse: {}".format(
    metrics.mean_squared_error(result_data, lsr_y_all_predict)))

# Elastic Net regression
enr = lm.ElasticNet(alpha=lsr_alpha,l1_ratio=enr_l1r)
enr_score = -cross_val_score(enr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
enr.fit(x_all_train, y_all_train)
enr_y_all_predict = ss_y.inverse_transform(enr.predict(x_all_train))
enr_y_test_predict = ss_y.inverse_transform(enr.predict(x_test))
print("enr_valid_mse: {}".format(enr_score.mean()))
print("enr_all_mse: {}".format(
    metrics.mean_squared_error(result_data, enr_y_all_predict)))

#%% save result
# result = proc_data[['student_ID','GPA']][proc_data['test_tag']=='test']
# result['GPA'] = svr_y_test_predict
# result.columns=['学生ID','综合GPA']
# insert_line = pd.DataFrame([['40dc29f67d3a0ea205e4',3.584083]],columns=['学生ID','综合GPA'])
# above_result = result[:58]
# below_result = result[58:]
# result = pd.concat([above_result,insert_line,below_result],ignore_index=True)
# result.to_csv('result/result_{}_svr.csv'.format(time.strftime("%b_%d_%H-%M-%S",time.localtime())),
#             header=True,index=False,encoding='utf-8')

result = proc_data[['student_ID','GPA']][proc_data['test_tag']=='test']
result['GPA'] = regr_y_test_predict
result.columns=['学生ID','综合GPA']
insert_line = pd.DataFrame([['40dc29f67d3a0ea205e4',fill_in_gpa]],columns=['学生ID','综合GPA'])
above_result = result[:58]
below_result = result[58:]
result = pd.concat([above_result,insert_line,below_result],ignore_index=True)
result.to_csv('result/result_{}_regr.csv'.format(time.strftime("%b_%d_%H-%M-%S",time.localtime())),
            header=True,index=False,encoding='utf-8')

result = proc_data[['student_ID','GPA']][proc_data['test_tag']=='test']
result['GPA'] = lsr_y_test_predict
result.columns=['学生ID','综合GPA']
insert_line = pd.DataFrame([['40dc29f67d3a0ea205e4',fill_in_gpa]],columns=['学生ID','综合GPA'])
above_result = result[:58]
below_result = result[58:]
result = pd.concat([above_result,insert_line,below_result],ignore_index=True)
result.to_csv('result/result_{}_lsr.csv'.format(time.strftime("%b_%d_%H-%M-%S",time.localtime())),
            header=True,index=False,encoding='utf-8')

result = proc_data[['student_ID','GPA']][proc_data['test_tag']=='test']
result['GPA'] = enr_y_test_predict
result.columns=['学生ID','综合GPA']
insert_line = pd.DataFrame([['40dc29f67d3a0ea205e4',fill_in_gpa]],columns=['学生ID','综合GPA'])
above_result = result[:58]
below_result = result[58:]
result = pd.concat([above_result,insert_line,below_result],ignore_index=True)
result.to_csv('result/result_{}_enr.csv'.format(time.strftime("%b_%d_%H-%M-%S",time.localtime())),
            header=True,index=False,encoding='utf-8')
