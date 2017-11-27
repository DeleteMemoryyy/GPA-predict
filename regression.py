# -*- coding: UTF-8 -*-
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn.preprocessing as prep
import sklearn.svm as svm
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
import seaborn as sns

# all_data = pd.ExcelFile('data/ALLDATA.xlsx').parse('Sheet1')

all_data = pd.read_csv('data/ALLDATA.csv')

drop_columns = ['grade','admit_grade','high_school','high_rank',
                'rank_var','progress','patent','social','prize','competition']

one_hot_columns = ['province','gender','birth_year','nation','politics','color_blind',
                   'stu_type','lan_type','sub_type','test_year','department','reward_type']

numerical_columns = ['left_sight','right_sight','admit_rank','school_num','center_grade'
                   'school_admit_rank','reward_score','height','weight']

standardization_columns = ['left_sight','right_sight','admit_rank','school_num',
                   'school_admit_rank','reward_score','height','weight']

other_columns = ['student_ID','GPA','test_tag','test_ID']

all_data['rank_var']=all_data['rank_var'].fillna(all_data['rank_var'].mean())
all_data['high_rank']=all_data['high_rank'].fillna(all_data['high_rank'].mean())
all_data['progress']=all_data['progress'].fillna(all_data['progress'].mean())
all_data['admit_grade']=all_data['admit_grade'].fillna(all_data['admit_grade'].mean())
all_data['center_grade'] = all_data['center_grade'].fillna(0)
all_data['reward_score'] = all_data['reward_score'].fillna(0)
all_data['prize']=all_data['prize'].fillna(0)
all_data['competition']=all_data['competition'].fillna(0)
all_data['patent']=all_data['patent'].fillna(0)
all_data['social']=all_data['social'].fillna(0)

proc_data = all_data

# one-hot processing
proc_data[one_hot_columns] = proc_data[one_hot_columns].fillna('Empty')
proc_data = pd.merge(proc_data, pd.get_dummies(proc_data, columns=one_hot_columns))
proc_data = proc_data.drop(one_hot_columns,axis=1)

# process sight
def precess_sight(x):
    if (x == np.nan):
        return x
    if(x>=3.0):
        return x
    elif (x>=2.0 and x<3.0):
            return 5.3
    elif (x>=1.5 and x<2.0):
        return 5.2
    elif (x>=1.2 and x<1.5):
        return 5.1
    elif (x>=1.0 and x<1.2):
        return 5.0
    elif (x>=1.0 and x<1.2):
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

proc_data['left_sight'] = proc_data['left_sight'].apply(precess_sight)
proc_data['right_sight'] = proc_data['right_sight'].apply(precess_sight)
proc_data['left_sight'] = proc_data['left_sight'].fillna(proc_data['left_sight'].mean())
proc_data['right_sight'] = proc_data['right_sight'].fillna(proc_data['right_sight'].mean())

# drop features
proc_data = proc_data.drop(drop_columns,axis=1)
proc_data = proc_data.drop(other_columns,axis=1)

#%% standardization
ss_x = prep.StandardScaler()
proc_data[standardization_columns] = ss_x.fit_transform(proc_data[standardization_columns].values)

#%% spilt training data
x_all_train = proc_data[all_data['test_tag']!='test']
y_all_train = all_data['GPA'][all_data['test_tag']!='test']
x_test = proc_data[all_data['test_tag']=='test']
x_train,x_valid,y_train,y_valid = train_test_split(x_all_train.values,y_all_train.values,
                                                   random_state=33)

#%% SVR
svr = svm.SVR(C=1000, gamma=0.001)
svr.fit(x_train, y_train)
svr_y_valid_predict = svr.predict(x_valid)
svr_y_test_predict = svr.predict(x_test)
result = all_data[['student_ID','GPA']][all_data['test_tag']=='test']
result['GPA'] = svr_y_test_predict
result.columns=['学生ID','综合GPA']
insert_line = pd.DataFrame([['40dc29f67d3a0ea205e4',3.584083]],columns=['学生ID','综合GPA'])
above_result = result[:58]
below_result = result[58:]
result = pd.concat([above_result,insert_line,below_result],ignore_index=True)
result.to_csv('result/result_{}.csv'.format(time.strftime("%b_%d_%H-%M-%S",time.localtime())),
              header=True,index=False,encoding='utf-8')

print("svr_score: {}".format(svr.score(x_valid,y_valid)))
print("MSE: {}".format(metrics.mean_squared_error(y_valid,svr_y_valid_predict)))
