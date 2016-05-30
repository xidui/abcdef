# -*- coding: utf-8 -*-
import math
import pandas as pd
import os
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn import linear_model
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

import warnings
warnings.filterwarnings("ignore")

def get_timeslot(day,timeid):
	return ['2016-01-'+str(day[i])+'-'+str(timeid[i])   for i in range(len(day))]


base_path = './'


def out_put_result():
	pass



f_path=base_path +"processed_data/train_data/train_day_30.csv"
df=pd.read_csv(f_path)
feasible_columns=df.columns

#read train data day8-14
path=base_path +"processed_data/train_data/train_combined.csv"
if os.path.exists(path):
	train=pd.read_csv(path)
else:
	flag=False
	for i in range(8,15):
		f_path=base_path +"processed_data/train_data/train_day_"+str(i )+".csv"
		df=pd.read_csv(f_path)
		if flag==False:
			train=df
			flag=True
		else:
			train=train.append(df)
	train.to_csv(path,index=False)

#read test data day15-21
path=base_path +"processed_data/train_data/test_combined.csv"
if os.path.exists(path):
	test=pd.read_csv(path)
else:
	flag=False
	for i in range(15,22):
		f_path=base_path +"processed_data/train_data/train_day_"+str(i )+".csv"
		df=pd.read_csv(f_path)
		if flag==False:
			test=df
			flag=True
		else:
			test=test.append(df)
	test.to_csv(path,index=False)





#read result data day22-30
path=base_path +"processed_data/train_data/result_combined.csv"
if os.path.exists(path):
	predict=pd.read_csv(path)
else:
	flag=False
	for i in [22,24,26,28,30]:
		f_path=base_path +"processed_data/train_data/train_day_"+str(i )+".csv"
		df=pd.read_csv(f_path)
		if flag==False:
			predict=df
			flag=True
		else:
			predict=predict.append(df)
	predict.to_csv(path,index=False)


#In2hour_tj_level_1	In2hour_tj_level_2	In2hour_tj_level_3	In2hour_tj_level_4


columns_to_drop = ['start_district_hash', 'In2hour_tj_level_1','In2hour_tj_level_2','In2hour_tj_level_3','In2hour_tj_level_4']
# feasible_columns = ["district_id", "Day", "Weekday", "Workday", "Yesterday_Workday", "time_id",
# 					"In30Min_Order_cnt", "In30Min_Supply_cnt", "In30Min_Gap_cnt", "In30Min_Twoday_ago_Order_cnt",
# 					"In30Min_Twoday_ago_Supply_cnt", "In30Min_Twoday_ago_Gap_cnt", "In30Min_AllDistrict_Order_cnt",
# 					"In30Min_AllDistrict_Supply_cnt", "In30Min_AllDistrict_Gap_cnt"]
#
# feasible_columns = feasible_columns + columns_to_drop


y=train.Gap_cnt.fillna(0)
y_test=test.Gap_cnt.fillna(0)
x=train[feasible_columns].drop(columns_to_drop, axis=1).fillna(0)
x_test=test[feasible_columns].drop(columns_to_drop, axis=1).fillna(0)
x_result=predict[feasible_columns].drop(columns_to_drop, axis=1).fillna(0)
x_result=x_result.loc[[i in [46,58,70,82,94,106,118,130,142] for i in x_result.time_id],:]
x_final=x_result.reset_index(drop=True)

a=x.columns
mapper=[]
for j in a:
	if j in ['district_id', 'Day', 'Weekday', 'Workday', 'Yesterday_Workday','Twoday_ago_Workday', 'time_id']:
		mapper.append((j,None))
	else:
		mapper.append((j,StandardScaler()))
b=DataFrameMapper(mapper)
b.fit(pd.concat([x, x_test, x_result]))
x=b.transform(x)
x_test=b.transform(x_test)
x_result_before = x_result
x_result=b.transform(x_result)

#Random Forest
clf = ensemble.RandomForestClassifier(n_estimators=20,max_features=min(len(feasible_columns) - len(columns_to_drop), 25))
clf.fit(x,y)
clf_predict=clf.predict(x_test)
clf_score=clf.score(x_test, y_test)


clf_predict.fill(1)

diff=clf_predict-y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "RandomForest MAPE: "+str(MAPE)

lower=[max(i-1,0) for i in clf_predict]
diff=lower - y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "RandomForest MAPE -1: "+str(MAPE)

upper=[max(i+0.5,0) for i in clf_predict]
diff=upper - y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "RandomForest MAPE +0.5: "+str(MAPE)

upper=[max(i+1,0) for i in clf_predict]
diff=upper - y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "RandomForest MAPE +1: "+str(MAPE)

upper=[min(i,30) for i in upper]
diff=upper-y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "RandomForest MAPE +1.5: "+str(MAPE)


'''
sel = SelectFromModel(clf, prefit=True)
x=sel.transform(x)
x_test=sel.transform(x_test)

clf = ensemble.RandomForestClassifier(n_estimators=20,max_features=15)
clf.fit(x,y)
clf_predict=clf.predict(x_test)
clf_score=clf.score(x_test, y_test)
'''



#Linear
clf = linear_model.LinearRegression()
clf.fit(x,y)
clf_predict=clf.predict(x_test)
clf_score=clf.score(x_test, y_test)

diff=clf_predict-y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "Linear MAPE: "+str(MAPE)


#Ridge
clf = linear_model.Ridge (alpha = .5)
clf.fit(x,y)
clf_predict=clf.predict(x_test)
clf_score=clf.score(x_test, y_test)

diff=clf_predict-y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "Ridge MAPE: "+str(MAPE)


#Lasso
clf = linear_model.Lasso(alpha = 0.1)
clf.fit(x,y)
clf_predict=clf.predict(x_test)
clf_score=clf.score(x_test, y_test)

diff=clf_predict-y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "Lasso MAPE: "+str(MAPE)




#LR L1
clf = linear_model.LogisticRegression(penalty='l1', tol=0.01)
clf.fit(x,y)
clf_predict=clf.predict(x_test)
clf_score=clf.score(x_test, y_test)

diff=clf_predict-y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "LR L1 MAPE: "+str(MAPE)


#LR L2
clf = linear_model.LogisticRegression(penalty='l2', tol=0.1)
clf.fit(x,y)
clf_predict=clf.predict(x_test)
clf_score=clf.score(x_test, y_test)

diff=clf_predict-y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "LR L2 MAPE: "+str(MAPE)


#SVM
import time

clf = svm.SVC()
print 'svm fit:' + time.time()
clf.fit(x,y)
print 'svm predict:' + time.time()
# clf_predict=clf.predict(x_test)
# clf_score=clf.score(x_test, y_test)
result_predict = clf.predict(x_result)
print 'svm done'
x_result_before['GAP'] = result_predict
f = open('./result6.csv', 'w')
for line in x_result_before[['district_id', 'Day', 'time_id', 'GAP']].values:
	f.write("{0},2016-01-{1}-{2},{3}\n".format(
		int(line[0]),
		int(line[1]),
		int(line[2]),
		float(line[3])
	))
f.close()

diff=clf_predict-y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/66/len(y_test))
print "SVM MAPE: "+str(MAPE)

#Gradient Boosting
params = {'n_estimators': 20, 'max_depth': 8, 'min_samples_split': 20,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(x,y)
clf_predict=clf.predict(x_test)
clf_score=clf.score(x_test, y_test)

diff=clf_predict-y_test
MAPE=sum(abs(diff[y_test!=0]/y_test[y_test!=0])/len(y_test))
print "Gradient Boosting MAPE: "+str(MAPE)


'''
#xgboost
y_result=clf.predict(x_result)
sample=pd.read_csv("sample result.csv",header=None )
sample.columns=['distinct_id','time_slot','prediction']
sample=sample.drop('prediction',axis=1)
output=pd.DataFrame(columns=['distinct_id','time_slot','prediction'])
output.distinct_id=x_final.district_id
output.time_slot=get_timeslot(x_final.Day,x_final.time_id)
output.prediction=y_result

final_output=pd.merge(sample,output,how='left')

final_output.to_csv("result.csv",index=False,header=False)
'''
