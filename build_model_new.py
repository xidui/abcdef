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

def try_modify_result(y_predict,y_test,model_name):
    diff = clf_predict - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test))
    print model_name+" MAPE: " + str(MAPE)

    lower = [max(i - 0.5, 1) for i in clf_predict]
    diff = lower - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test))
    print model_name+" MAPE -0.5: " + str(MAPE)

    lower = [max(i - 1, 1) for i in clf_predict]
    diff = lower - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test))
    print model_name+" MAPE -1: " + str(MAPE)

    lower = [max(i - 2, 1) for i in clf_predict]
    diff = lower - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test))
    print model_name+" MAPE -2: " + str(MAPE)

    upper = [max(i - 3, 1) for i in clf_predict]
    diff = upper - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test))
    print model_name+" MAPE -3: " + str(MAPE)

    upper = [max(i + 0.5, 1) for i in clf_predict]
    diff = upper - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test))
    print model_name+" MAPE +0.5: " + str(MAPE)

    upper = [max(i + 1, 0) for i in clf_predict]
    diff = upper - y_test
    MAPE = sum(abs(diff[y_test != 0] / y_test[y_test != 0]) / len(y_test))
    print model_name+" MAPE +1: " + str(MAPE)




def get_feathers(mother_table):
    #cluster
    mother_table['is_cluster_0'] = [i in [7, 48, 51] for i in mother_table.district_id]
    mother_table['is_cluster_1'] = [i in [1, 12, 14, 20, 24, 27, 28, 37, 42, 46, 8, 23] for i in mother_table.district_id]
    mother_table['is_cluster_2'] = [i in [2, 3, 4, 5, 6, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66] for i in mother_table.district_id]
    # rush-hour
    mother_table['is_morning_rushhour'] = [i in range(46, 55) for i in mother_table.time_id]
    mother_table['is_noon_rushhour'] = [i in range(102, 112) for i in mother_table.time_id]
    mother_table['is_night_rushhour'] = [i in range(122, 135) for i in mother_table.time_id]
    #weekends
    mother_table['is_monday'] = [i==2 for i in mother_table.time_id]
    mother_table['is_friday'] = [i==4 for i in mother_table.time_id]
    return mother_table
'''
# 是否应该在这里先填0？？ 是否所有的列都填0？填1？
    mother_table=mother_table.fillna(0)

#用第一周的数据会好一些？
    son_table=mother_table['start_district_hash','Order_cnt'].groupby(['start_district_hash']).mean().reset_index().rename(columns={'Order_cnt':'district_average_Order'})
    mother_table=pd.merge(mother_table,son_table,on=['start_district_hash'],how='left')
    son_table=mother_table['start_district_hash','Supply_cnt'].groupby(['start_district_hash']).mean().reset_index().rename(columns={'Supply_cnt':'district_average_Supply'})
    mother_table=pd.merge(mother_table,son_table,on=['start_district_hash'],how='left')
    son_table=mother_table['start_district_hash','Gap_cnt'].groupby(['start_district_hash']).mean().reset_index().rename(columns={'Gap_cnt':'district_average_Gap'})
    mother_table=pd.merge(mother_table,son_table,on=['start_district_hash'],how='left')

#cut-off selection   How to get best cut-off?
    mother_table['is_high_Order']=[i >=60  for i in mother_table.district_average_Order]
    mother_table['is_medium_Order']=[i in range(7,60)  for i in mother_table.district_average_Order]
    mother_table['is_low_Order']=[i <7  for i in mother_table.district_average_Order]
    mother_table['is_high_Gap']=[i >=30  for i in mother_table.district_average_Gap]
    mother_table['is_medium_Gap']=[i in range(3,30)  for i in mother_table.district_average_Gap]
    mother_table['is_low_Gap']=[i <3  for i in mother_table.district_average_Gap]
'''

'''
    son_table=mother_table[mother_table.is_morning_rushhour==1]['start_district_hash','Workday','Order_cnt'].groupby(['start_district_hash','Workday'])
    .mean().reset_index().rename(columns={'Order_cnt':'morning_rushhour_workday_mean_Order'})
    mother_table=pd.merge(mother_table,son_table,how='left')
    son_table=mother_table[mother_table.is_morning_rushhour==1]['start_district_hash','Workday','Supply_cnt'].groupby(['start_district_hash','Workday'])
    .mean().reset_index().rename(columns={'Supply_cnt':'morning_rushhour_workday_mean_Supply'})
    mother_table=pd.merge(mother_table,son_table,how='left')
    son_table=mother_table[mother_table.is_morning_rushhour==1]['start_district_hash','Workday','Gap_cnt'].groupby(['start_district_hash','Workday'])
    .mean().reset_index().rename(columns={'Gap_cnt':'morning_rushhour_workday_mean_Gap'})
    mother_table=pd.merge(mother_table,son_table,how='left')

    son_table=mother_table[mother_table.is_morning_rushhour==1]['start_district_hash','Weekday','Order_cnt'].groupby(['start_district_hash','Weekday'])
    .mean().reset_index().rename(columns={'Order_cnt':'morning_rushhour_weekday_mean_Order'})
    mother_table=pd.merge(mother_table,son_table,how='left')
    son_table=mother_table[mother_table.is_morning_rushhour==1]['start_district_hash','Weekday','Supply_cnt'].groupby(['start_district_hash','Weekday'])
    .mean().reset_index().rename(columns={'Supply_cnt':'morning_rushhour_weekday_mean_Supply'})
    mother_table=pd.merge(mother_table,son_table,how='left')
    son_table=mother_table[mother_table.is_morning_rushhour==1]['start_district_hash','Weekday','Gap_cnt'].groupby(['start_district_hash','Weekday'])
    .mean().reset_index().rename(columns={'Gap_cnt':'morning_rushhour_weekday_mean_Gap'})
    mother_table=pd.merge(mother_table,son_table,how='left')
    
    son_table=mother_table[mother_table.is_noon_rushhour==1]['start_district_hash','Workday','Order_cnt'].groupby(['start_district_hash','Workday'])
    .mean().reset_index().rename(columns={'Order_cnt':'morning_rushhour_workday_mean_Order'})
    mother_table=pd.merge(mother_table,son_table,how='left')
    son_table=mother_table[mother_table.is_noon_rushhour==1]['start_district_hash','Workday','Supply_cnt'].groupby(['start_district_hash','Workday'])
    .mean().reset_index().rename(columns={'Supply_cnt':'morning_rushhour_workday_mean_Supply'})
    mother_table=pd.merge(mother_table,son_table,how='left')
    son_table=mother_table[mother_table.is_noon_rushhour==1]['start_district_hash','Workday','Gap_cnt'].groupby(['start_district_hash','Workday'])
    .mean().reset_index().rename(columns={'Gap_cnt':'morning_rushhour_workday_mean_Gap'})
    mother_table=pd.merge(mother_table,son_table,how='left')

    son_table=mother_table[mother_table.is_noon_rushhour==1]['start_district_hash','Weekday','Order_cnt'].groupby(['start_district_hash','Weekday'])
    .mean().reset_index().rename(columns={'Order_cnt':'morning_rushhour_weekday_mean_Order'})
    mother_table=pd.merge(mother_table,son_table,how='left')
    son_table=mother_table[mother_table.is_noon_rushhour==1]['start_district_hash','Weekday','Supply_cnt'].groupby(['start_district_hash','Weekday'])
    .mean().reset_index().rename(columns={'Supply_cnt':'morning_rushhour_weekday_mean_Supply'})
    mother_table=pd.merge(mother_table,son_table,how='left')
    son_table=mother_table[mother_table.is_noon_rushhour==1]['start_district_hash','Weekday','Gap_cnt'].groupby(['start_district_hash','Weekday'])
    .mean().reset_index().rename(columns={'Gap_cnt':'morning_rushhour_weekday_mean_Gap'})
    mother_table=pd.merge(mother_table,son_table,how='left')
'''
    


    
    
    
f_path=base_path +"processed_data/train_data/train_day_22.csv"
df=pd.read_csv(f_path)
feasible_columns=df.columns

#read train data day8-14
path=base_path +"processed_data/train_data/train_combined.csv"
if os.path.exists(path):
    train_all=pd.read_csv(path)
else:
    flag=False
    for i in range(8,15):
        f_path=base_path +"processed_data/train_data/train_day_"+str(i )+".csv"
        df=pd.read_csv(f_path)
        if flag==False:
            train_all=df
            flag=True
        else:
            train_all=train_all.append(df)
    train_all.to_csv(path,index=False)

#read test data day15-21
path=base_path +"processed_data/train_data/test_combined.csv"
if os.path.exists(path):
    test_all=pd.read_csv(path)
else:
    flag=False
    for i in range(15,22):
        f_path=base_path +"processed_data/train_data/train_day_"+str(i )+".csv"
        df=pd.read_csv(f_path)
        if flag==False:
            test_all=df
            flag=True
        else:
            test_all=test_all.append(df)
    test_all.to_csv(path,index=False)





#read result data day22-30
path=base_path +"processed_data/train_data/result_combined.csv"
if os.path.exists(path):
    predict_all=pd.read_csv(path)
else:
    flag=False
    for i in [22,24,26,28,30]:
        f_path=base_path +"processed_data/train_data/train_day_"+str(i )+".csv"
        df=pd.read_csv(f_path)
        if flag==False:
            predict_all=df
            flag=True
        else:
            predict_all=predict_all.append(df)
    predict_all.to_csv(path,index=False)


#train_all=get_feathers(train_all)
#test_all=get_feathers(test_all)
#predict_all=get_feathers(predict_all)
    

columns_to_drop = ['start_district_hash', 'In2hour_tj_level_1','In2hour_tj_level_2','In2hour_tj_level_3','In2hour_tj_level_4']

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
settings=[2,1]
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
clust = 'is_cluster_' + str(settings[0])
train=train_all[train_all.Workday==settings[1]]
train=train[train[clust]==1]
test=test_all[test_all.Workday==settings[1]]
test=test[test[clust]==1]
predict=predict_all[predict_all.Workday==settings[1]]
predict=predict[predict[clust]==1]
output_file="./result/r_"+str(settings[0])+"_"+str(settings[1])+".csv"


y=train.Gap_cnt.fillna(1)
#y=[math.log(i+1,10) for i in y]
y_test=test.Gap_cnt.fillna(0)
#y_test=[math.log(i+1,10) for i in y_test]
x=train[feasible_columns].drop(columns_to_drop, axis=1).fillna(0)
x_test=test[feasible_columns].drop(columns_to_drop, axis=1).fillna(0)
x_result=predict[feasible_columns].drop(columns_to_drop, axis=1).fillna(0)
x_result=x_result.loc[[i in [46,58,70,82,94,106,118,130,142] for i in x_result.time_id],:]
x_final=x_result.reset_index(drop=True)

a=x.columns
mapper=[]
for j in a:
    if j in ['district_id', 'Day', 'Weekday', 'Workday', 'Yesterday_Workday','Twoday_ago_Workday', 'time_id',
    'Weather','weather_1','weather_2','weather_3','weather_4','weather_5','weather_6','weather_7','weather_8','weather_9']:
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
clf = ensemble.RandomForestClassifier(n_estimators=20,max_features=min(len(feasible_columns) - len(columns_to_drop), 20))
clf.fit(x,y)
clf_predict=clf.predict(x_test)
clf_score=clf.score(x_test, y_test)


clf_predict.fill(1)
try_modify_result(clf_predict,y_test,"Random Forest")

'''
y_result=clf.predict(x_result)
y_result=[max(i+1,1) for i in y_result]
sample=pd.read_csv("sample result.csv",header=None )
sample.columns=['distinct_id','time_slot','prediction']
sample=sample.drop('prediction',axis=1)
output=pd.DataFrame(columns=['distinct_id','time_slot','prediction'])
output.distinct_id=x_final.district_id
output.time_slot=get_timeslot(x_final.Day,x_final.time_id)
output.prediction=y_result

final_output=pd.merge(sample,output,on=['distinct_id','time_slot'],how='left')

final_output.to_csv(output_file,index=False,header=False)
print len(output)
print len(final_output)

'''
#Gradient Boosting
params = {'n_estimators': 40, 'max_depth': 8, 'min_samples_split': 20,
          'learning_rate': 0.01, 'loss': 'lad'}
clf = ensemble.GradientBoostingRegressor(**params)

#weight = [40/math.log10(i+10) for i in y]

clf.fit(x,y)
clf_predict=clf.predict(x_test)
clf_score=clf.score(x_test, y_test)
try_modify_result(clf_predict,y_test,"Gradient Boosting")

y_result=clf.predict(x_result)
#y_result=[max(i-3,1) for i in y_result]

sample=pd.read_csv("sample result.csv",header=None )
sample.columns=['distinct_id','time_slot','prediction']
sample=sample.drop('prediction',axis=1)
output=pd.DataFrame(columns=['distinct_id','time_slot','prediction'])
output.distinct_id=x_final.district_id
output.time_slot=get_timeslot(x_final.Day,x_final.time_id)
output.prediction=y_result

final_output=pd.merge(sample,output,on=['distinct_id','time_slot'],how='left')

final_output.to_csv(output_file,index=False)
print len(output)
print len(final_output)



sel = SelectFromModel(clf, prefit=True)
x=sel.transform(x)
x_test=sel.transform(x_test)



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


'''

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
'''

#xgboost