# -*- coding: UTF-8 -*-
import time
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn import preprocessing as prep
from sklearn import linear_model as lm
from sklearn import svm as svm
from sklearn import metrics as metrics
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import warnings

# plot setting
# %matplotlib inline
warnings.filterwarnings('ignore')
sns.set(style='white', color_codes=True)
myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)
sns.set(font=myfont.get_name())


# all_data = pd.ExcelFile('data/ALLDATA.xlsx').parse('Sheet1')
all_data = pd.read_csv('data/ALLDATA.csv')
dpart_rank = pd.read_csv('data/depart_rank.csv')
all_data = pd.concat([all_data, dpart_rank], axis=1)

svr_C = 400
svr_gamma = 0.001
regr_alpha = 11.0
lsr_alpha = 0.00062
enr_alpha = 0.001086
enr_l1r = 0.5

rand_seed = 33

all_columns = ['student_ID', 'province', 'gender', 'birth_year', 'nation', 'politics',
 'left_sight', 'right_sight', 'color_blind', 'height', 'weight',
 'stu_type', 'lan_type', 'sub_type', 'test_year', 'high_school', 'grade',
 'admit_grade', 'admit_rank', 'center_grade', 'school_num',
 'school_admit_rank', 'department', 'reward_score', 'reward_type',
 'high_rank', 'rank_var', 'progress', 'patent', 'social', 'prize',
 'competition', 'GPA', 'test_tag', 'test_ID', 'dpart_rank']

ori_one_hot_columns = ['province', 'gender', 'test_year', 'nation','politics','color_blind', 'stu_type', 'lan_type', 'sub_type', 'birth_year','department','reward_type']

ori_numerical_columns = ['left_sight', 'right_sight', 'height', 'weight', 'grade',
                         'admit_grade', 'admit_rank', 'center_grade', 'dpart_rank', 'reward_score', 'school_num', 'school_admit_rank',
                         'high_rank', 'rank_var', 'progress', 'patent', 'social', 'prize',
                         'competition']

drop_columns = ['grade', 'admit_grade', 'high_school', 'politics',
                'color_blind', 'lan_type', 'left_sight', 'right_sight', 'patent']

one_hot_columns = ['province', 'gender', 'birth_year', 'nation',
                   'test_year', 'stu_type', 'sub_type', 'department', 'reward_type']

numerical_columns = ['admit_rank', 'school_num', 'center_grade',
                     'school_admit_rank', 'dpart_rank', 'reward_score', 'competition', 'height', 'weight', 'social', 'prize', 'high_rank', 'rank_var', 'progress']

standardization_columns = ['admit_rank', 'school_num',
                           'school_admit_rank', 'dpart_rank', 'reward_score', 'competition', 'height', 'weight']

other_columns = ['student_ID', 'GPA', 'test_tag', 'test_ID']

# preprocess features
# drop outlier
for i in range(all_data.shape[0]):
    if(all_data['test_tag'][i] != 'test' and all_data['GPA'][i] <= 0.3):
        all_data = all_data.drop(i, axis=0)
all_data.index
all_data.index = range(all_data.shape[0])

# fill nan
all_data['rank_var'] = all_data['rank_var'].fillna(all_data['rank_var'].mean())
all_data['high_rank'] = all_data['high_rank'].fillna(
    all_data['high_rank'].mean())
all_data['progress'] = all_data['progress'].fillna(all_data['progress'].mean())
all_data['grade'] = all_data['grade'].fillna(all_data['grade'].mean())
all_data['admit_grade'] = all_data['admit_grade'].fillna(
    all_data['admit_grade'].mean())
all_data['center_grade'] = all_data['center_grade'].fillna(0)
all_data['reward_score'] = all_data['reward_score'].fillna(0)
all_data['prize'] = all_data['prize'].fillna(0)
all_data['competition'] = all_data['competition'].fillna(0)
all_data['patent'] = all_data['patent'].fillna(0)
all_data['social'] = all_data['social'].fillna(0)

# process birth_year
def process_birth_year(x):
    test_age = x['test_year'] - x['birth_year']
    if (test_age <= 15):
        test_age = 15
    elif (test_age >= 20):
        test_age = 20
    return test_age
all_data['birth_year'] = all_data.apply(process_birth_year,axis=1)

# process high_rank
for i in range(all_data.shape[0]):
    if(all_data['high_rank'][i] >= 0.5):
        all_data['high_rank'][i] /= 350.0

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
all_data['left_sight'] = all_data['left_sight'].apply(process_sight)
all_data['right_sight'] = all_data['right_sight'].apply(process_sight)
all_data['left_sight'] = all_data['left_sight'].fillna(
    all_data['left_sight'].mean())
all_data['right_sight'] = all_data['right_sight'].fillna(
    all_data['right_sight'].mean())

# process nation
d_nation = {}
temp_nation = []
for i in range(all_data.shape[0]):
    if (all_data['nation'][i] in d_nation.keys()):
            d_nation[all_data['nation'][i]] += 1
    else:
        d_nation[all_data['nation'][i]] = 1
for i in range(all_data.shape[0]):
    if (d_nation[all_data['nation'][i]] <= 6):
        temp_nation.append('少数民族')
    else:
        temp_nation.append(all_data['nation'][i])
all_data['nation'] = temp_nation

train_data = all_data[all_data['test_tag']!='test']

# for col in all_data.columns:
#     if (col in ori_one_hot_columns):
#         sns.boxplot(x=col,y='GPA',data=train_data)
#         sns.stripplot(x=col, y='GPA', data=train_data, jitter=True)
#         plt.show()
#     elif (col in ori_numerical_columns):
#         sns.jointplot(x=col,y='GPA',data=train_data,kind='reg')
#         plt.show()

d_re = {}
for col in ori_one_hot_columns:
    d_re[col] = prep.LabelEncoder()
    train_data[col] = d_re[col].fit_transform(train_data[col])
school = prep.StandardScaler()
train_data[ori_numerical_columns] = school.fit_transform(train_data[ori_numerical_columns].values)

x_all_train = pd.concat([train_data[ori_one_hot_columns],train_data[ori_numerical_columns]],ignore_index=True,axis=1)
y_all_train = train_data['GPA']

# rfr = ensemble.RandomForestRegressor(oob_score=True)
# grid = GridSearchCV(rfr, param_grid={'n_estimators':range(10,101,10)},cv=5)
# grid.fit(x_all_train,y_all_train)
# print('Best n_estimators: {}'.format(grid.best_params_['n_estimators']))

rfr = ensemble.RandomForestRegressor(n_estimators=10,oob_score=True)
rfr_score = -cross_val_score(rfr, x_all_train,
                             y_all_train, cv=5, scoring='neg_mean_squared_error')
rfr.fit(x_all_train, y_all_train)
rfr_y_all_predict = rfr.predict(x_all_train)
print("rfr_valid_mse: {}".format(rfr_score.mean()))
print("rfr_all_mse: {}".format(
    metrics.mean_squared_error(y_all_train, rfr_y_all_predict)))

fi_idx = ori_one_hot_columns
fi_idx.extend(ori_numerical_columns)
feature_importances = pd.DataFrame([rfr.feature_importances_,fi_idx]).transpose()
feature_importances.columns = ['feature_importances','features']
sns.factorplot(data=feature_importances,x='feature_importances',y='features',kind='bar')
plt.show()
