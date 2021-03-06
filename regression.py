# -*- coding: UTF-8 -*-
#%% init
import time
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing as prep
from sklearn import ensemble
from sklearn import neural_network
from sklearn import linear_model as lm
from sklearn import kernel_ridge
from sklearn import svm as svm
from sklearn import tree as tree
from sklearn import metrics
import rgf.sklearn as rgf
import xgboost as xgb
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None

# proc_data = pd.ExcelFile('data/ALLDATA.xlsx').parse('Sheet1')
all_data = pd.read_csv('data/ALLDATA.csv')
dpart_rank = pd.read_csv('data/depart_rank.csv')
center_progress = pd.read_csv('data/center_progress.csv')
center_rank = pd.read_csv('data/center_rank.csv')
center_var = pd.read_csv('data/center_var.csv')
center_competition = pd.read_csv('data/center_competition.csv')
center_reward = pd.read_csv('data/center_reward.csv')
center_social = pd.read_csv('data/center_social.csv')
BMI = pd.read_csv('data/BMI.csv')
all_data = pd.concat([all_data, dpart_rank, center_progress, center_rank,
                      center_var, center_competition, center_reward, center_social, BMI], axis=1)
proc_data = all_data

drop_gpa = 0.5

svr_C = 200
svr_gamma = 0.001
regr_alpha = 11.0
lsr_alpha = 0.0005547
enr_alpha = 0.0009649
enr_l1r = 0.5
gbr_n_estimators = 400
rfr_n_estimators = 90
krr_alpha = 0.6
krr_coef0 = 2.5
krr_degree = 4
krr_gamma = 0.001
stacker_regr_alpha = 11.0
stacker_enr_alpha = 0.00009649
stacker_enr_l1r = 0.5
stacker_weight_regr = 1.0
stacker_weight_enr = 0.0
stacker_weight_krr = 0.0
stacker_weight_lgbr = 0.0

rand_seed = 314
fill_in_gpa = 2.35815726

ori_one_hot_columns = ['province', 'gender', 'test_year', 'nation', 'politics', 'color_blind',
                       'stu_type', 'lan_type', 'sub_type', 'birth_year', 'department', 'reward_type']

ori_numerical_columns = ['left_sight', 'right_sight', 'height', 'weight', 'BMI', 'grade', 'center_progress', 'center_rank','center_var','admit_grade', 'admit_rank', 'center_grade', 'dpart_rank', 'reward_score', 'school_num', 'school_admit_rank','high_rank', 'rank_var', 'progress', 'patent', 'social', 'prize','competition']

drop_columns = ['grade', 'admit_grade', 'high_rank', 'rank_var', 'progress',  'center_rank',
                'center_progress', 'center_var', 'color_blind', 'lan_type', 'left_sight', 'right_sight', 'patent', 'BMI']

one_hot_columns = ['province', 'politics', 'gender', 'birth_year', 'nation', 'test_year', 'stu_type', 'sub_type', 'department', 'reward_type']

numerical_columns = ['admit_rank', 'school_num', 'center_grade', 'social','center_social',
                     'school_admit_rank', 'dpart_rank', 'reward_score', 'center_reward', 'competition', 'center_competition', 'height', 'weight', 'high_school']

other_columns = ['student_ID', 'GPA', 'test_tag', 'test_ID']

# preprocess features
# drop outlier
for i in range(proc_data.shape[0]):
    if(proc_data['test_tag'][i] != 'test' and proc_data['GPA'][i] <= drop_gpa):
        proc_data = proc_data.drop(i, axis=0)
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

# process high_school
proc_data['high_school'] = prep.LabelEncoder().fit_transform(proc_data['high_school'])

# process birth_year
def process_birth_year(x):
    test_age = x['test_year'] - x['birth_year']
    if test_age <= 15:
        test_age = 15
    elif test_age >= 20:
        test_age = 20
    return test_age
proc_data['birth_year'] = proc_data.apply(process_birth_year, axis=1)

# process height and weight
male_height_mean = proc_data['height'][proc_data['gender'] == '男'].mean()
male_height_std = proc_data['height'][proc_data['gender'] == '男'].std()
female_height_mean = proc_data['height'][proc_data['gender'] == '女'].mean()
female_height_std = proc_data['height'][proc_data['gender'] == '女'].std()
proc_data['height'][proc_data['gender'] == '男'] = (proc_data['height'][proc_data['gender'] == '男']-male_height_mean)/male_height_std
proc_data['height'][proc_data['gender'] == '女'] = (
    proc_data['height'][proc_data['gender'] == '女'] - female_height_mean) / female_height_std
male_weight_mean = proc_data['weight'][proc_data['gender'] == '男'].mean()
male_weight_std = proc_data['weight'][proc_data['gender'] == '男'].std()
female_weight_mean = proc_data['weight'][proc_data['gender'] == '女'].mean()
female_weight_std = proc_data['weight'][proc_data['gender'] == '女'].std()
proc_data['weight'][proc_data['gender'] == '男'] = (
    proc_data['weight'][proc_data['gender'] == '男'] - male_weight_mean) / male_weight_std
proc_data['weight'][proc_data['gender'] == '女'] = (
    proc_data['weight'][proc_data['gender'] == '女'] - female_weight_mean) / female_weight_std

# process sight
def process_sight(x):
    if x == np.nan:
        return x
    if x >= 3.0:
        return x
    elif x >= 2.0 and x < 3.0:
            return 5.3
    elif x >= 1.5 and x < 2.0:
        return 5.2
    elif x >= 1.2 and x < 1.5:
        return 5.1
    elif x >= 1.0 and x < 1.2:
        return 5.0
    elif x >= 1.0 and x < 1.2:
        return 5.0
    elif x >= 0.8 and x < 1.0:
        return 4.9
    elif x >= 0.6 and x < 0.8:
        return 4.8
    elif x >= 0.5 and x < 0.6:
        return 4.7
    elif x >= 0.4 and x < 0.5:
        return 4.6
    elif x >= 0.3 and x < 0.4:
        return 4.5
    elif x >= 0.25 and x < 0.3:
        return 4.4
    elif x >= 0.2 and x < 0.25:
        return 4.3
    elif x >= 0.15 and x < 0.2:
        return 4.2
    elif x >= 0.12 and x < 0.15:
        return 4.1
    elif x >= 0.1 and x < 0.12:
        return 4.0
    elif x >= 0.08 and x < 0.1:
        return 3.9
    elif x >= 0.08 and x < 0.1:
        return 3.9
    elif x >= 0.06 and x < 0.08:
        return 3.8
    elif x >= 0.05 and x < 0.06:
        return 3.7
    elif x >= 0.04 and x < 0.05:
        return 3.6
    elif x >= 0.03 and x < 0.04:
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
    if proc_data['nation'][i] in d_nation.keys():
        d_nation[proc_data['nation'][i]] += 1
    else:
        d_nation[proc_data['nation'][i]] = 1
for i in range(proc_data.shape[0]):
    if d_nation[proc_data['nation'][i]] <= 6:
        temp_nation.append('少数民族')
    else:
        temp_nation.append(proc_data['nation'][i])
proc_data['nation'] = temp_nation

# process high_rank
def process_high_rank(x):
    temp_high_rank = x['high_rank']
    if temp_high_rank >= 0.5:
        temp_high_rank /= 350
    return temp_high_rank
proc_data['high_rank'] = proc_data.apply(process_high_rank,axis=1)

# one-hot processing
proc_data[one_hot_columns] = proc_data[one_hot_columns].fillna('Empty')
proc_data = pd.get_dummies(proc_data, columns=one_hot_columns)

# drop features
proc_data = proc_data.drop(drop_columns,axis=1)

# standardization
ss_x = prep.StandardScaler()
proc_data[numerical_columns] = ss_x.fit_transform(proc_data[numerical_columns].values)

# spilt training data
x_all_train = proc_data[proc_data['test_tag']!='test'].drop(other_columns,axis=1)
x_test = proc_data[proc_data['test_tag'] == 'test'].drop(other_columns, axis=1)
result_data = proc_data['GPA'][proc_data['test_tag']!='test']
y_all_train = result_data.values

#%% regression
svr = svm.SVR(C=svr_C, gamma=svr_gamma)
regr = lm.Ridge(alpha=regr_alpha)
lsr = lm.Lasso(alpha=lsr_alpha)
enr = lm.ElasticNet(alpha=lsr_alpha,l1_ratio=enr_l1r)
krr = kernel_ridge.KernelRidge(kernel='polynomial')
gbr = ensemble.GradientBoostingRegressor(
    loss='huber', max_features='sqrt', n_estimators=gbr_n_estimators)
rfr = ensemble.RandomForestRegressor(n_estimators=rfr_n_estimators)
mlpr = neural_network.MLPRegressor(hidden_layer_sizes=(256,256),learning_rate='invscaling',learning_rate_init=0.01,max_iter=300,random_state=rand_seed,early_stopping=True,verbose=False)
xgbr = xgb.XGBRegressor(booster='gbtree',gamma=0.001, max_depth=3, min_child_weight=2, n_estimators=100)
xgblr = xgb.XGBRegressor(booster='gblinear',n_estimators=300,gamma=0.0001)
lgbr = lgb.LGBMRegressor(num_leaves=3,min_data_in_leaf=11,max_bin=55,learning_rate=0.05,n_estimators=900)
rgfr = rgf.RGFRegressor(max_leaf=700,learning_rate=0.005,min_samples_leaf=5,test_interval=50)
class Stacking(object):
    def __init__(self, n_folds, stackers, stacker_weight, base_models):
        self.n_folds = n_folds
        self.stackers = stackers
        self.stacker_weight = stacker_weight
        self.base_models = base_models

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rand_seed)

        s_train = np.zeros((X.shape[0], len(self.base_models)))

        for i, mod in enumerate(self.base_models):
            j = 0
            for idx_train, idx_valid in kf.split(range(len(y))):
                x_train_j = X[idx_train]
                y_train_j = y[idx_train]
                x_valid_j = X[idx_valid]

                mod.fit(x_train_j, y_train_j)

                y_valid_j = mod.predict(x_valid_j)[:]
                s_train[idx_valid, i] = y_valid_j

                j += 1

        for stacker in self.stackers:
            stacker.fit(s_train, y)

    def predict(self,T):
        T = np.array(T)
        s_test = np.zeros((T.shape[0], len(self.base_models)))
        y_predict = np.zeros((T.shape[0],len(self.stackers)))
        y_predict_weighted = np.zeros((T.shape[0],))

        for i, mod in enumerate(self.base_models):
            s_test[:, i] = mod.predict(T)[:]

        for i, stacker in enumerate(self.stackers):
            y_predict[:, i] = stacker.predict(s_test)[:]
            y_predict_weighted += y_predict[:, i] * self.stacker_weight[i]

        return y_predict_weighted

    def fit_predict(self,X,y,T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rand_seed)

        s_train = np.zeros((X.shape[0], len(self.base_models)))
        s_test = np.zeros((T.shape[0], len(self.base_models)))
        y_predict = np.zeros((T.shape[0], len(self.stackers)))
        y_predict_weighted = np.zeros((T.shape[0],))

        for i, mod in enumerate(self.base_models):
            s_test_i = np.zeros((s_test.shape[0], kf.get_n_splits()))

            j = 0
            for idx_train, idx_valid in kf.split(range(len(y))):
                x_train_j = X[idx_train]
                y_train_j = y[idx_train]
                x_valid_j = X[idx_valid]

                mod.fit(x_train_j, y_train_j)

                y_valid_j = mod.predict(x_valid_j)[:]
                s_train[idx_valid, i] = y_valid_j
                s_test_i[:, j] = mod.predict(T)[:]

                j += 1

            s_test[:, i] = s_test_i.mean(1)

        for stacker in self.stackers:
            stacker.fit(s_train, y)

        for i, stacker in enumerate(self.stackers):
            y_predict[:, i] = stacker.predict(s_test)[:]
            y_predict_weighted += y_predict[:, i] * self.stacker_weight[i]

        return y_predict_weighted

# #%% dimensionality reduction
# rfr.fit(x_all_train,y_all_train)
# colum_name = x_all_train.columns
# feature_importances = rfr.feature_importances_
# sorted_feature = sorted(zip(map(lambda x: round(x, 4), feature_importances), colum_name),reverse=False)
# for i in range(6):
#     x_all_train = x_all_train.drop(sorted_feature[i][1],axis=1)
#     x_test = x_test.drop(sorted_feature[i][1],axis=1)

#%% 5-fold stacking
stacking = Stacking(n_folds=5, stackers=[lm.Ridge(alpha=stacker_regr_alpha), lgb.LGBMRegressor(num_leaves=3, min_data_in_leaf=11, max_bin=55, learning_rate=0.05,
                                                                                               n_estimators = 900)], stacker_weight=[stacker_weight_regr, stacker_weight_lgbr], base_models=[enr, regr, svr, gbr, rfr, krr, xgbr, xgblr, lgbr, rgfr])
folds = KFold(n_splits=5, shuffle=True, random_state=rand_seed).split(range(x_all_train.shape[0]))
stacking_score = []
for idx_train, idx_valid in folds:
    X = np.array(x_all_train)
    y = np.array(y_all_train)
    x_train = X[idx_train]
    x_valid = X[idx_valid]
    y_train = y[idx_train]
    y_valid = y[idx_valid]
    # stacking.fit(x_train,y_train)
    # stacking_y_valid_predict = stacking.predict(x_valid)
    stacking_y_valid_predict = stacking.fit_predict(x_train,y_train,x_valid)
    stacking_score.append(metrics.mean_squared_error(y_valid, stacking_y_valid_predict))
stacking_score = np.array(stacking_score)
print('stacking_valid_mse: {}'.format(stacking_score.mean()))
print('stacking_valid_mse_std: {}'.format(stacking_score.std()))
# stacking.fit(x_all_train, y_all_train)
# stacking_y_all_predict = stacking.predict(x_all_train)
stacking_y_test_predict = stacking.fit_predict(x_all_train, y_all_train, x_test)
# print('stacking_all_mse: {}'.format(
#     metrics.mean_squared_error(y_all_train, stacking_y_all_predict)))
# stacking_y_test_predict = stacking.predict(x_test)

#%% save result
result = proc_data[['student_ID','GPA']][proc_data['test_tag']=='test']
result['GPA'] = stacking_y_test_predict
result.columns=['学生ID','综合GPA']
insert_line = pd.DataFrame([['40dc29f67d3a0ea205e4',fill_in_gpa]],columns=['学生ID','综合GPA'])
above_result = result[:58]
below_result = result[58:]
result = pd.concat([above_result,insert_line,below_result],ignore_index=True)
save_name = 'result/result_{}_stacking.csv'.format(time.strftime('%b_%d_%H-%M-%S',time.localtime()))
result.to_csv(save_name,header=True,index=False,encoding='utf-8')
print('save to {}\n'.format(save_name))

# feature_importances = pd.DataFrame(
#     [stacking.base_models[4].feature_importances_, x_all_train.columns]).transpose()
# feature_importances.columns = ['feature_importances','features']
# sns.factorplot(data=feature_importances,x='feature_importances',y='features',kind='bar')
# plt.savefig('img/feature_importances_{}.jpg'.format(time.strftime('%b_%d_%H-%M-%S', time.localtime())))

# #%% weight
# stacker_regr_alpha = 11.0
# stacker_weight_regr = 0.8
# stacker_weight_lgbr = 0.2
# stacking = Stacking(n_folds=5, stackers=[lm.Ridge(alpha=stacker_regr_alpha), lgb.LGBMRegressor(num_leaves=3, min_data_in_leaf=11, max_bin=55, learning_rate=0.05,
#                                                                                                n_estimators=900)], stacker_weight=[stacker_weight_regr, stacker_weight_lgbr], base_models=[enr, regr, lsr, svr, gbr, rfr, krr, mlpr, xgbr, lgbr])
# folds = KFold(n_splits=5, shuffle=True, random_state=rand_seed).split(
#     range(x_all_train.shape[0]))
# idx_train, idx_valid = folds.__next__()
# X = np.array(x_all_train)
# y = np.array(y_all_train)
# x_train = X[idx_train]
# x_valid = X[idx_valid]
# y_train = y[idx_train]
# y_valid = y[idx_valid]
# stacking.fit(x_train,y_train)
# for t_weight_regr in range(1,11):
#     weight_regr = t_weight_regr/10.0
#     weight_lgbr = 1.0 - weight_regr
#     stacking.stacker_weight = [weight_regr, weight_lgbr]
#     stacking_y_valid_predict = stacking.predict(x_valid)
#     print('regr: {0:.3f}, lgbr: {1:.3f}, mse : {2:.10f}'.format(weight_regr, weight_lgbr,
#                                                                metrics.mean_squared_error(y_valid, stacking_y_valid_predict)))
# print('---------------------------')

#%% Lasso regression
# lsr_score = -cross_val_score(lsr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
# lsr.fit(x_all_train, y_all_train)
# lsr_y_all_predict = lsr.predict(x_all_train)
# lsr_y_test_predict = lsr.predict(x_test)
# print('lsr_valid_mse: {}'.format(lsr_score.mean()))
# print('lsr_all_mse: {}'.format(
#     metrics.mean_squared_error(result_data, lsr_y_all_predict)))

# #%% Ridge regression
# regr_score = -cross_val_score(regr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
# regr.fit(x_all_train, y_all_train)
# regr_y_all_predict = regr.predict(x_all_train)
# regr_y_test_predict = regr.predict(x_test)
# print('regr_valid_mse: {}'.format(regr_score.mean()))
# print('regr_all_mse: {}'.format(
#     metrics.mean_squared_error(result_data, regr_y_all_predict)))

# #%% Elastic Net regression
# enr_score = -cross_val_score(enr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
# enr.fit(x_all_train, y_all_train)
# enr_y_all_predict = enr.predict(x_all_train)
# enr_y_test_predict = enr.predict(x_test)
# print('enr_valid_mse: {}'.format(enr_score.mean()))
# print('enr_all_mse: {}'.format(
#     metrics.mean_squared_error(result_data, enr_y_all_predict)))

# #%% Kernel Ridge regression
# krr_score = -cross_val_score(krr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
# krr.fit(x_all_train, y_all_train)
# krr_y_all_predict = krr.predict(x_all_train)
# krr_y_test_predict = krr.predict(x_test)
# print('krr_valid_mse: {}'.format(krr_score.mean()))
# print('krr_all_mse: {}'.format(
#     metrics.mean_squared_error(result_data, krr_y_all_predict)))

# #%% SVR
# svr_score = -cross_val_score(svr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
# svr.fit(x_all_train, y_all_train)
# svr_y_all_predict = svr.predict(x_all_train)
# svr_y_test_predict = svr.predict(x_test)
# print('svr_valid_mse: {}'.format(svr_score.mean()))
# print('svr_all_mse: {}'.format(metrics.mean_squared_error(result_data,svr_y_all_predict)))

# #%% GBR
# gbr_score = -cross_val_score(gbr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
# gbr.fit(x_all_train, y_all_train)
# gbr_y_all_predict = gbr.predict(x_all_train)
# gbr_y_test_predict = gbr.predict(x_test)
# print('gbr_valid_mse: {}'.format(gbr_score.mean()))
# print('gbr_all_mse: {}'.format(metrics.mean_squared_error(result_data,gbr_y_all_predict)))

# #%% XGBR
# xgbr_score = -cross_val_score(xgbr,x_all_train,y_all_train,cv=5,scoring='neg_mean_squared_error')
# xgbr.fit(x_all_train, y_all_train)
# xgbr_y_all_predict = xgbr.predict(x_all_train)
# xgbr_y_test_predict = xgbr.predict(x_test)
# print('xgbr_valid_mse: {}'.format(xgbr_score.mean()))
# print('xgbr_all_mse: {}'.format(metrics.mean_squared_error(result_data,xgbr_y_all_predict)))

# #%% XGBLR
# xgblr_score = -cross_val_score(xgblr, x_all_train,
#                               y_all_train, cv=5, scoring='neg_mean_squared_error')
# xgblr.fit(x_all_train, y_all_train)
# xgblr_y_all_predict = xgblr.predict(x_all_train)
# xgblr_y_test_predict = xgblr.predict(x_test)
# print('xgblr_valid_mse: {}'.format(xgblr_score.mean()))
# print('xgblr_all_mse: {}'.format(
#     metrics.mean_squared_error(result_data, xgblr_y_all_predict)))

# #%% LightGBM
# lgbr_score = -cross_val_score(lgbr, x_all_train,
#                                y_all_train, cv=5, scoring='neg_mean_squared_error')
# lgbr.fit(x_all_train, y_all_train)
# lgbr_y_all_predict = lgbr.predict(x_all_train)
# lgbr_y_test_predict = lgbr.predict(x_test)
# print('lgbr_valid_mse: {}'.format(lgbr_score.mean()))
# print('lgbr_all_mse: {}'.format(
#     metrics.mean_squared_error(result_data, lgbr_y_all_predict)))

# #%% Random Forest
# rfr_score = -cross_val_score(rfr, x_all_train,
#                                y_all_train, cv=5, scoring='neg_mean_squared_error')
# rfr.fit(x_all_train, y_all_train)
# rfr_y_all_predict = rfr.predict(x_all_train)
# rfr_y_test_predict = rfr.predict(x_test)
# print('rfr_valid_mse: {}'.format(rfr_score.mean()))
# print('rfr_all_mse: {}'.format(
#     metrics.mean_squared_error(result_data, rfr_y_all_predict)))

# #%% Regularized Greedy Forest
# rgfr_score = -cross_val_score(rgfr, x_all_train,
#                              y_all_train, cv=5, scoring='neg_mean_squared_error')
# rgfr.fit(x_all_train, y_all_train)
# rgfr_y_all_predict = rgfr.predict(x_all_train)
# rgfr_y_test_predict = rgfr.predict(x_test)
# print('rgfr_valid_mse: {}'.format(rgfr_score.mean()))
# print('rgfr_all_mse: {}'.format(
#     metrics.mean_squared_error(result_data, rgfr_y_all_predict)))
