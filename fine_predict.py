# -*- coding: UTF-8 -*-
from sklearn.ensemble import GradientBoostingRegressor
from math import log
import pandas as pd
import numpy as np 

ALL=pd.read_csv('./Data.csv')

_STRING_COLUMNS=[
	'hometown','gender','birth_year','nation','politics','color_blind','stu_type','lan_type','test_year',
	'high_school','test_type','department','city','admit_type','sub_type']

_NUMERIC_COLUMNS=[
	'left_sight','right_sight','height','weight','grade','admit_grade','high_rank','rank_var','progress',
	'patent','social','prize','competition']

#preprocessing of the numeric columns
school_member=ALL['high_school'].value_counts()
ALL['school_member']=0


ALL['rank_var']=ALL['rank_var'].fillna(ALL['rank_var'].mean())
ALL['high_rank']=ALL['high_rank'].fillna(ALL['high_rank'].mean())
ALL['progress']=ALL['progress'].fillna(ALL['progress'].mean())
ALL['admit_grade']=ALL['admit_grade'].fillna(ALL['admit_grade'].mean())
ALL['prize']=ALL['prize'].fillna(0)
ALL['competition']=ALL['competition'].fillna(0)
ALL['patent']=ALL['patent'].fillna(0)
ALL['social']=ALL['social'].fillna(0)

for i in range(ALL.shape[0]):
	ALL['school_member'][i]=school_member[ALL['high_school'][i]]
	if(ALL['rank_var'][i]==0):
		ALL['rank_var'][i]=ALL['rank_var'].mean()
	if(ALL['high_rank'][i]==0):
		ALL['high_rank'][i]=ALL['high_rank'].mean()
	if(ALL['progress'][i]==0):
		ALL['progress'][i]=ALL['progress'].mean()



for name in _STRING_COLUMNS:
	ALL[name]=ALL[name].fillna('')
	ALL[name]=pd.get_dummies(ALL[name])

ALL=ALL.drop('high_school',1)
ALL=ALL.drop('studentid',1)
ALL=ALL.drop('left_sight',1)
ALL=ALL.drop('right_sight',1)
ALL=ALL.drop('grade',1)
ALL=ALL.drop('color_blind',1)
ALL=ALL.drop('politics',1)
ALL=ALL.drop('city',1)
ALL=ALL.drop('progress',1)
ALL=ALL.drop('height',1)
#ALL=ALL.drop('rank_var',1)


Train=ALL[ALL['mark']!='test']
Test=ALL[ALL['mark']=='test']

Pre=Train[Train['mark']!='pre']
Valid=Train[Train['mark']=='pre']

Train=Train.drop('mark',1)
Pre=Pre.drop('mark',1)
Valid=Valid.drop('mark',1)
Test=Test.drop('mark',1)

Train_label=Train['gpa']
Pre_label=Pre['gpa']
Valid_label=Valid['gpa']

Train=Train.drop('gpa',1)
Pre=Pre.drop('gpa',1)
Valid=Valid.drop('gpa',1)
Test=Test.drop('gpa',1)


model=GradientBoostingRegressor(learning_rate=0.01,n_estimators=500,warm_start=True)

#semi_Pre=[Pre['department'],Pre['admit_grade'],Pre['hometown']]
#semi_Valid=[Valid['department'],Valid['admit_grade'],Valid['hometown']]


model.fit(Pre,Pre_label)


y=model.predict(Valid)

print(((y-Valid_label)**2).mean())


#model.fit(Train,Train_label)

yy=model.predict(Test)

np.savetxt('./predict.csv',yy,delimiter=',')




