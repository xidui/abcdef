# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
import os
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

pic_cnt=0
f_path="./training_data/poi_data/poi.csv"
df=pd.read_csv(f_path).fillna(0)
'''
f_path="./processed_data/train_data/train_day_8.csv"
table=pd.read_csv(f_path)
for i in [11,12,13,14,15,18,19,20,21]:
    f_path="./processed_data/train_data/train_day_"+str(i)+".csv"
    table=table.append(pd.read_csv(f_path))
table.to_csv("./processed_data/train_data/workday.csv")

f_path="./processed_data/train_data/train_day_9.csv"
table=pd.read_csv(f_path)
for i in [10,16,17]:
    f_path="./processed_data/train_data/train_day_"+str(i)+".csv"
    table=table.append(pd.read_csv(f_path))
table.to_csv("./processed_data/train_data/weekend.csv")
'''
table=pd.read_csv("./processed_data/train_data/workday.csv")

m=table[['start_district_hash','district_id']].drop_duplicates().reset_index(drop=True)

df=pd.merge(df,m,on='start_district_hash',how='left').sort_values(by=['district_id']).drop(['start_district_hash'],axis=1)
m=m.drop(['start_district_hash'],axis=1)

poi_train=df.drop(['district_id'],axis=1)
poi_train=StandardScaler().fit(poi_train).transform(poi_train)

n_clusters=3


'''
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(poi_train)
labels = kmeans.labels_


poi_labes=labels
print "Based on poi:"
for k in range(n_clusters):
    distr=[]
    for i in range(len(labels)):
        if labels[i]==k:
            distr.append(i+1)
    print "cluster "+str(k)
    print distr
    b=table[[i in distr for i in table.district_id]]
    c=b[['time_id','Order_cnt','Gap_cnt']].groupby(['time_id']).mean().reset_index()
    
    plt.figure(pic_cnt)
    pic_cnt=pic_cnt+1
    plt.plot(c.time_id,c.Order_cnt,'g')
    plt.plot(c.time_id,c.Gap_cnt,'r')
    plt.gca().set_xticks(np.linspace(43,145,18)) 
    plt.show()
'''

for t in range(43,145):
    if (t+2)%3==0:
        son=table[[i in range(t,t+3) for i in table.time_id]]
        son=son[['district_id','Order_cnt','Gap_cnt']].groupby(['district_id']).mean().reset_index()
        son=son.rename(columns={'Order_cnt':'Order_mean_'+str(t),'Gap_cnt':'Gap_mean_'+str(t+3)})
        m=pd.merge(m,son,on='district_id',how='left')
'''
for t in range(43,145):
    if (t+2)%3==0:
        son=table[[i in range(t,t+3) for i in table.time_id]]
        son=son[['district_id','Gap_cnt']].groupby(['district_id']).mean().reset_index()
        son=son.rename(columns={'Gap_cnt':'Gap_mean_'+str(t)})
        m=pd.merge(m,son,on='district_id',how='left')
'''
m=m.fillna(0)

gap_train=m.drop(['district_id'],axis=1)
gap_train=StandardScaler().fit(gap_train).transform(gap_train)
'''
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(gap_train)
labels = kmeans.labels_
print "Based on order & gap:"
for k in range(n_clusters):
    distr=[]
    for i in range(len(labels)):
        if labels[i]==k:
            distr.append(i+1)
    print "cluster "+str(k)
    print distr
    b=table[[i in distr for i in table.district_id]]
    c=b[['time_id','Order_cnt','Gap_cnt']].groupby(['time_id']).mean().reset_index()
    
    plt.figure(pic_cnt)
    pic_cnt=pic_cnt+1
    plt.plot(c.time_id,c.Order_cnt,'g')
    plt.plot(c.time_id,c.Gap_cnt,'r')
    plt.gca().set_xticks(np.linspace(43,145,18)) 
    plt.show()

gap_labes=labels
'''


aa=pd.merge(df,m,on='district_id',how='left')

all_train=aa.drop(['district_id'],axis=1)
all_train=StandardScaler().fit(all_train).transform(all_train)

kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(all_train)
labels = kmeans.labels_
print "Based on poi & order & gap:"
for k in range(n_clusters):
    distr=[]
    for i in range(len(labels)):
        if labels[i]==k:
            distr.append(i+1)
    print "cluster "+str(k)
    print distr
    b=table[[i in distr for i in table.district_id]]
    c=b[['time_id','Order_cnt','Gap_cnt']].groupby(['time_id']).mean().reset_index()
    
    plt.figure(pic_cnt)
    pic_cnt=pic_cnt+1
    plt.plot(c.time_id,c.Order_cnt,'g')
    plt.plot(c.time_id,c.Gap_cnt,'r')
    plt.gca().set_xticks(np.linspace(43,145,18)) 
    plt.show()

all_labes=labels


'''
weekends

Based on poi:
cluster 0
[2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
cluster 1
[51]
cluster 2
[1, 7, 8, 12, 23, 27, 28, 37, 48]
Based on order & gap:
cluster 0
[2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
cluster 1
[1, 7, 8, 23, 24, 28, 37, 46, 48]
cluster 2
[51]
Based on poi & order & gap:
cluster 0
[2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
cluster 1
[51]
cluster 2
[1, 7, 8, 12, 23, 24, 28, 37, 46, 48]




weekdays

Based on poi:
cluster 0
[1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
cluster 1
[7, 8, 12, 23, 48]
cluster 2
[51]
Based on order & gap:
cluster 0
[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
cluster 1
[7, 8, 23, 28, 37, 46, 48]
cluster 2
[51]
Based on poi & order & gap:
cluster 0
[2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
cluster 1
[51]
cluster 2
[1, 7, 8, 12, 23, 28, 37, 46, 48]





Based on poi & order & gap:
cluster 0
[2, 3, 4, 5, 6, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
cluster 1
[7, 48]
cluster 2
[51]
cluster 3
[1, 12, 14, 20, 24, 27, 28, 37, 42, 46]
cluster 4
[8, 23]
'''



'''
cluster 0
7, 48, 51

cluster 1
1, 12, 14, 20, 24, 27, 28, 37, 42, 46, 8, 23

cluster 2
2, 3, 4, 5, 6, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66
'''