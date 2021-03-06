# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import time

base_path = './'

def get_time_id(timestamp):
	t = timestamp.split()[1].split(':')
	return int(t[0]) * 6 + int(t[1]) / 10 + 1


def get_timestamp_by_id(id):
	return "{0:0>2}:{1:0>2}:00".format(id / 6, (id % 6) * 10)


def minute_transfer(minute):
	m=int(minute)
	if m<10 :
		r='00'
	elif m<20:
		r='10'
	elif m<30:
		r='20'
	elif m<40:
		r='30'
	elif m<50:
		r='40'
	else:
		r='50'
	return r


def get_key_table():
	#table contains the basic keys
	#区域定义表
	action_file=base_path + 'training_data/cluster_map/cluster_map'
	b=pd.read_table(action_file,header=None)
	b=pd.DataFrame(b.get_values(),columns=['district_hash','district_id'])
	b=b.rename(columns={'district_hash':'start_district_hash'})

	mother_table=b.drop_duplicates()
	mother_table['commen_key']=1
	son_table=pd.DataFrame(range(1, 22),columns=['Day'])
	son_table['Weekday']=[4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3]
	#son_table['Weekday']=[4,5,6,7,1,2,3,4,5]
	son_table['Workday']=[1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1]
	#son_table['Workday']=[1,0,0,1,1,1,1,1,0]
	son_table['Yesterday_Workday']=[1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1]
	#son_table['Yesterday_Workday']=[1,1,0,0,1,1,1,1,1]
	son_table['Twoday_ago_Workday']=[1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1]
	#son_table['Twoday_ago_Workday']=[1,1,1,0,0,1,1,1,1]
	son_table['commen_key']=1
	mother_table=pd.merge(mother_table,son_table,how='left')

	son_table=pd.DataFrame(range(1,145),columns=['time_id'])
	son_table['commen_key']=1
	mother_table=pd.merge(mother_table,son_table,how='left')
	mother_table=mother_table.drop(['commen_key'],axis=1)
	mother_table.to_csv(base_path + 'processed_data/test_test.csv',index=False)
	#could save this table every time appended a attribute
	return mother_table


def Order_Supply_Gap_attribute(a,mother_table,keys,name_header):
	son_table=a.groupby(keys).count().reset_index()[range(0,len(keys)+1)]
	rename=son_table.columns[-1]
	son_table=son_table.rename(columns={rename:name_header+'Order_cnt'})
	mother_table=pd.merge(mother_table,son_table,on=keys,how='left')

	son_table=a[a.driver_id.notnull()].groupby(keys).count().reset_index()[range(0,len(keys)+1)]
	rename=son_table.columns[-1]
	son_table=son_table.rename(columns={rename:name_header+'Supply_cnt'})
	mother_table=pd.merge(mother_table,son_table,on=keys,how='left')

	son_table=a[a.driver_id.isnull()].groupby(keys).count().reset_index()[range(0,len(keys)+1)]
	rename=son_table.columns[-1]
	son_table=son_table.rename(columns={rename:name_header+'Gap_cnt'})
	mother_table=pd.merge(mother_table,son_table,on=keys,how='left')
	return mother_table

def price_attribute(b,mother_table,columns,keys,name_header):
	b=b[columns]
	son_table=b.groupby(keys).mean().reset_index().rename(columns={'Price':name_header+'Avg_Price'})
	mother_table=pd.merge(mother_table,son_table,on=keys,how='left')

	son_table=b.groupby(keys).std().reset_index().rename(columns={'Price':name_header+'Std_Price'})
	mother_table=pd.merge(mother_table,son_table,on=keys,how='left')

	son_table=b.groupby(keys).max().reset_index().rename(columns={'Price':name_header+'Max_Price'})
	mother_table=pd.merge(mother_table,son_table,on=keys,how='left')

	son_table=b.groupby(keys).min().reset_index().rename(columns={'Price':name_header+'Min_Price'})
	mother_table=pd.merge(mother_table,son_table,on=keys,how='left')
	return mother_table

def distinct_driver_attribute(b,mother_table,keys_with_driver,keys,name_header):
	b=b.groupby(keys_with_driver).count().reset_index()[range(0,len(keys_with_driver))]
	son_table=b.groupby(keys).count().reset_index().rename(columns={'driver_id':name_header+'Distinct_Driver'})
	mother_table=pd.merge(mother_table,son_table,on=keys,how='left')
	return mother_table

def order_data_processing(a):
	a.Time=[i[:17]+'00'  for i in a.Time]
	a['Day']=[int(i[8:10])  for i in a.Time]
	a['Hour']=[int(i[11:13]) for i in a.Time]
	a['Minute']=[int(minute_transfer(i[14:16]))  for i in a.Time]
	a['Weekday']=[int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday())  for i in a.Time]
	a['time_id']=[get_time_id(i)  for i in a.Time]
	a.Price=[float(i)  for i in a.Price]
	return a


# primary key for each sample
key_table_path=base_path + 'processed_data/test_test.csv'
if os.path.exists(key_table_path):
	key_table=pd.read_csv(key_table_path)
else:
	key_table=get_key_table()



# now get attribute for day 8 , only use data in day 1-8

for processing_day in [22, 24, 26, 28, 30]:
	mother_table=key_table[key_table.Day==processing_day]
	mother_table=mother_table[[i in range(43,145) for i in mother_table.time_id]]
	a=pd.DataFrame(columns=['order_id','driver_id','passenger_id',
		'start_district_hash','dest_district_hash','Price','Time'])


	#--------------------------------------------------------------------------------------------------------------------


	print "start get 30 minute today attributes"
	# order data needed
	fileday=processing_day
	if fileday<10:
		action_file=base_path + "training_data/order_data/order_data_2016-01-0"+str(fileday)
	else:
		action_file=base_path + "training_data/order_data/order_data_2016-01-"+str(fileday)

	a = pd.read_table(action_file,header=None)
	a = pd.DataFrame(a.get_values(),columns=['order_id','driver_id','passenger_id',
	'start_district_hash','dest_district_hash','Price','Time'])
	a = order_data_processing(a)


	#Important !!!  target value Order_Supply_Gap
	if processing_day in range(8, 22):
		mother_table=Order_Supply_Gap_attribute(a,mother_table,['start_district_hash','Day','time_id'],'')


	#attributes in 30 minute today
	b=pd.DataFrame(columns=a.columns)
	for now_time_id in range(43,145):
		temp_table=a[[tmp in range(now_time_id-3,now_time_id) for tmp in a.time_id]]
		temp_table.loc[:,['time_id']]=now_time_id
		b=b.append(temp_table)

	mother_table=Order_Supply_Gap_attribute(b,mother_table,['start_district_hash','Day','time_id'],'In30Min_')
	mother_table=Order_Supply_Gap_attribute(b,mother_table,['Day','time_id'],'In30Min_AllDistrict_')

	b_not_null=b[b.driver_id.notnull()]
	mother_table=price_attribute(b_not_null,mother_table,['start_district_hash','Day','time_id','Price'],['start_district_hash','Day','time_id'],'In30Min_Day_')
	mother_table=price_attribute(b_not_null,mother_table,['Day','time_id','Price'],['Day','time_id'],'In30Min_AllDistrict_Day_')

	mother_table=distinct_driver_attribute(b_not_null,mother_table,['Day','time_id','driver_id'],['Day','time_id'],'In30Min_AllDistrict_')


	#--------------------------------------------------------------------------------------------------------------------

	print "start get Twoday_ago attributes"
	# attributes in Twoday_ago    only in 30 minutes
	fileday=processing_day-2
	if fileday<10:
		action_file=base_path + "training_data/order_data/order_data_2016-01-0"+str(fileday)
	else:
		action_file=base_path + "training_data/order_data/order_data_2016-01-"+str(fileday)

	if os.path.exists(action_file):
		temp_a=pd.read_table(action_file,header=None)
		temp_a=pd.DataFrame(temp_a.get_values(),columns=['order_id','driver_id','passenger_id',
		'start_district_hash','dest_district_hash','Price','Time'])
		a = temp_a
		a = order_data_processing(a)

		#attributes in 30 minute Twoday_ago
		b=pd.DataFrame(columns=a.columns)
		for now_time_id in range(43,145):
			temp_table=a[[tmp in range(now_time_id-3,now_time_id) for tmp in a.time_id]]
			temp_table.loc[:,['time_id']]=now_time_id
			temp_table.loc[:,['Day']]=processing_day
			b=b.append(temp_table)

		mother_table=Order_Supply_Gap_attribute(b,mother_table,['start_district_hash','Day','time_id'],'In30Min_Twoday_ago_')
		mother_table=Order_Supply_Gap_attribute(b,mother_table,['Day','time_id'],'In30Min_AllDistrict_Twoday_ago_')

		b_not_null=b[b.driver_id.notnull()]
		mother_table=price_attribute(b_not_null,mother_table,['start_district_hash','Day','time_id','Price'],['start_district_hash','Day','time_id'],'In30Min_Twoday_ago_')
		mother_table=price_attribute(b_not_null,mother_table,['Day','time_id','Price'],['Day','time_id'],'In30Min_AllDistrict_Twoday_ago_')
		mother_table=distinct_driver_attribute(b_not_null,mother_table,['Day','time_id','driver_id'],['Day','time_id'],'In30Min_AllDistrict_Twoday_ago_')


	#--------------------------------------------------------------------------------------------------------------------

	print "start get last weekday attributes"
	# attributes in last weekday
	fileday=processing_day-7
	if fileday<10:
		action_file=base_path + "training_data/order_data/order_data_2016-01-0"+str(fileday)
	else:
		action_file=base_path + "training_data/order_data/order_data_2016-01-"+str(fileday)
	if os.path.exists(action_file):
		temp_a=pd.read_table(action_file,header=None)
		temp_a=pd.DataFrame(temp_a.get_values(),columns=['order_id','driver_id','passenger_id',
		'start_district_hash','dest_district_hash','Price','Time'])
		a=temp_a
		a=order_data_processing(a)


		mother_table=Order_Supply_Gap_attribute(a,mother_table,['start_district_hash','Weekday','time_id'],'LastWeekday_')
		mother_table=price_attribute(a,mother_table,['start_district_hash','Weekday','time_id','Price'],['start_district_hash','Weekday','time_id'],'Weekday_')
		mother_table=price_attribute(a,mother_table,['Weekday','time_id','Price'],['Weekday','time_id'],'AllDistrict_Weekday_')

		#attributes in 30 minute last weekday
		b=pd.DataFrame(columns=a.columns)
		for now_time_id in range(43,145):
			temp_table=a[[tmp in range(now_time_id-3,now_time_id) for tmp in a.time_id]]
			temp_table.loc[:,['time_id']]=now_time_id
			b=b.append(temp_table)

		mother_table=Order_Supply_Gap_attribute(b,mother_table,['start_district_hash','Weekday','time_id'],'In30Min_Weekday_')
		mother_table=Order_Supply_Gap_attribute(b,mother_table,['Weekday','time_id'],'In30Min_AllDistrict_Weekday_')

		b_not_null=b[b.driver_id.notnull()]
		mother_table=price_attribute(b_not_null,mother_table,['start_district_hash','Weekday','time_id','Price'],['start_district_hash','Weekday','time_id'],'In30Min_Weekday_')
		mother_table=price_attribute(b_not_null,mother_table,['Weekday','time_id','Price'],['Weekday','time_id'],'In30Min_AllDistrict_Weekday_')
		mother_table=distinct_driver_attribute(b_not_null,mother_table,['Weekday','time_id','driver_id'],['Weekday','time_id'],'In30Min_AllDistrict_Weekday_')


	#无人接单处理：重复下单，跨年，低价格、高价格、价格是否带小数-打折？





	#POI信息表
	poi_file=base_path + 'training_data/poi_data/poi.csv'
	if os.path.exists(poi_file):
		poi_table=pd.read_csv(poi_file)
	else:
		action_file=base_path + 'training_data/poi_data/poi_data'
		c= open(action_file).read().split('\n')
		c=c[:66]
		poi=[]
		poi_unique=[]
		cluster_hash_list=[]
		for i in c:
			q=i.split('\t')
			cluster_hash_list.append(q[0])
			poi_list=[]
			for j in q[1:]:
				poi_list.append(j.split(':'))
			poi.append(poi_list)
			for k in poi_list[1:]:
				if k[0] not in poi_unique:
					poi_unique.append(k[0])

		q=pd.DataFrame(poi_unique,columns=['poi'])
		q['level_1']=[int(i.split('#')[0]) for i in q.poi]
		tmp=[]
		for i in q.poi:
			if len(i.split('#'))==1:
				tmp.append(0)
			else:
				tmp.append(int(i.split('#')[1]))
		q['level_2']=tmp
		q=q.sort_values(by=['level_1','level_2'])


		column_name=[i for i in q.poi]
		column_name.insert(0,'cluster_hash')

		poi_table=pd.DataFrame(columns=column_name)
		poi_table['cluster_hash']=cluster_hash_list
		for i in range(0,len(poi)):
			for j in poi[i]:
				poi_table[j[0]][i]=int(j[1])/83
		poi_table=poi_table.fillna(0).rename(columns={'cluster_hash':'start_district_hash'})
		poi_table.to_csv(poi_file,index=False)
	mother_table=pd.merge(mother_table,poi_table,on=['start_district_hash'],how='left')

	#拥堵信息表
	#today
	fileday=processing_day
	if fileday<10:
		action_file=base_path + "processed_data/traffic_data/traffic_data_2016-01-0"+str(fileday)
	else:
		action_file=base_path + "processed_data/traffic_data/traffic_data_2016-01-"+str(fileday)

	if os.path.exists(action_file):
		d=pd.read_table(action_file,header=None)
		d=pd.DataFrame(d.get_values(),columns=['district_hash','tj_level_1','tj_level_2',
		'tj_level_3','tj_level_4','tj_time','time_id'])

		d.tj_level_1=[int(i.split(':')[1]) for i in d.tj_level_1]
		d.tj_level_2=[int(i.split(':')[1]) for i in d.tj_level_2]
		d.tj_level_3=[int(i.split(':')[1]) for i in d.tj_level_3]
		d.tj_level_4=[int(i.split(':')[1]) for i in d.tj_level_4]
		d.tj_time=processing_day
		#10min ago
		temp=d
		temp.time_id=temp.time_id+1
		temp=temp.rename(columns={'tj_time':'Day'})
		temp=temp.rename(columns={'district_hash':'start_district_hash'})
		for i in range(1,5):
			temp=temp.rename(columns={'tj_level_'+str(i):'In10min_tj_level_'+str(i)})
		mother_table=pd.merge(mother_table,temp,on=['start_district_hash','Day','time_id'],how='left')
		#30min ago
		temp=d
		temp.time_id=temp.time_id+3
		temp=temp.rename(columns={'tj_time':'Day'})
		temp=temp.rename(columns={'district_hash':'start_district_hash'})
		for i in range(1,5):
			temp=temp.rename(columns={'tj_level_'+str(i):'In30min_tj_level_'+str(i)})
		mother_table=pd.merge(mother_table,temp,on=['start_district_hash','Day','time_id'],how='left')
		#2h10min ago
		temp=d
		temp.time_id=temp.time_id+13
		temp=temp.rename(columns={'tj_time':'Day'})
		temp=temp.rename(columns={'district_hash':'start_district_hash'})
		for i in range(1,5):
			temp=temp.rename(columns={'tj_level_'+str(i):'In2hour_tj_level_'+str(i)})
		mother_table=pd.merge(mother_table,temp,on=['start_district_hash','Day','time_id'],how='left')


	#two_days_ago
	fileday=processing_day-2
	if fileday<10:
		action_file=base_path + "processed_data/traffic_data/traffic_data_2016-01-0"+str(fileday)
	else:
		action_file=base_path + "processed_data/traffic_data/traffic_data_2016-01-"+str(fileday)

	if os.path.exists(action_file):
		d=pd.read_table(action_file,header=None)
		d=pd.DataFrame(d.get_values(),columns=['district_hash','tj_level_1','tj_level_2',
		'tj_level_3','tj_level_4','tj_time','time_id'])

		d.tj_level_1=[int(i.split(':')[1]) for i in d.tj_level_1]
		d.tj_level_2=[int(i.split(':')[1]) for i in d.tj_level_2]
		d.tj_level_3=[int(i.split(':')[1]) for i in d.tj_level_3]
		d.tj_level_4=[int(i.split(':')[1]) for i in d.tj_level_4]
		d.tj_time=processing_day
		#10min ago
		temp=d
		temp.time_id=temp.time_id+1
		temp=temp.rename(columns={'tj_time':'Day'})
		temp=temp.rename(columns={'district_hash':'start_district_hash'})
		for i in range(1,5):
			temp=temp.rename(columns={'tj_level_'+str(i):'twodaysago_In10min_tj_level_'+str(i)})
		mother_table=pd.merge(mother_table,temp,on=['start_district_hash','Day','time_id'],how='left')
		#30min ago
		temp=d
		temp.time_id=temp.time_id+3
		temp=temp.rename(columns={'tj_time':'Day'})
		temp=temp.rename(columns={'district_hash':'start_district_hash'})
		for i in range(1,5):
			temp=temp.rename(columns={'tj_level_'+str(i):'twodaysago_In30min_tj_level_'+str(i)})
		mother_table=pd.merge(mother_table,temp,on=['start_district_hash','Day','time_id'],how='left')
		#2h10min ago
		temp=d
		temp.time_id=temp.time_id+13
		temp=temp.rename(columns={'tj_time':'Day'})
		temp=temp.rename(columns={'district_hash':'start_district_hash'})
		for i in range(1,5):
			temp=temp.rename(columns={'tj_level_'+str(i):'twodaysago_In2hour_tj_level_'+str(i)})
		mother_table=pd.merge(mother_table,temp,on=['start_district_hash','Day','time_id'],how='left')


	#last_weekday_ago
	fileday=processing_day-7
	if fileday<10:
		action_file=base_path + "processed_data/traffic_data/traffic_data_2016-01-0"+str(fileday)
	else:
		action_file=base_path + "processed_data/traffic_data/traffic_data_2016-01-"+str(fileday)

	if os.path.exists(action_file):
		d=pd.read_table(action_file,header=None)
		d=pd.DataFrame(d.get_values(),columns=['district_hash','tj_level_1','tj_level_2',
		'tj_level_3','tj_level_4','tj_time','time_id'])

		d.tj_level_1=[int(i.split(':')[1]) for i in d.tj_level_1]
		d.tj_level_2=[int(i.split(':')[1]) for i in d.tj_level_2]
		d.tj_level_3=[int(i.split(':')[1]) for i in d.tj_level_3]
		d.tj_level_4=[int(i.split(':')[1]) for i in d.tj_level_4]
		d.tj_time=processing_day
		#10min ago
		temp=d
		temp.time_id=temp.time_id+1
		temp=temp.rename(columns={'tj_time':'Day'})
		temp=temp.rename(columns={'district_hash':'start_district_hash'})
		for i in range(1,5):
			temp=temp.rename(columns={'tj_level_'+str(i):'Weekday_In10min_tj_level_'+str(i)})
		mother_table=pd.merge(mother_table,temp,on=['start_district_hash','Day','time_id'],how='left')
		#30min ago
		temp=d
		temp.time_id=temp.time_id+3
		temp=temp.rename(columns={'tj_time':'Day'})
		temp=temp.rename(columns={'district_hash':'start_district_hash'})
		for i in range(1,5):
			temp=temp.rename(columns={'tj_level_'+str(i):'Weekday_In30min_tj_level_'+str(i)})
		mother_table=pd.merge(mother_table,temp,on=['start_district_hash','Day','time_id'],how='left')
		#2h10min ago
		temp=d
		temp.time_id=temp.time_id+13
		temp=temp.rename(columns={'tj_time':'Day'})
		temp=temp.rename(columns={'district_hash':'start_district_hash'})
		for i in range(1,5):
			temp=temp.rename(columns={'tj_level_'+str(i):'Weekday_In2hour_tj_level_'+str(i)})
		mother_table=pd.merge(mother_table,temp,on=['start_district_hash','Day','time_id'],how='left')




	#天气信息表
	#today
	fileday=processing_day
	if fileday<10:
		action_file=base_path + "processed_data/weather_data/weather_data_2016-01-0"+str(fileday)
	else:
		action_file=base_path + "processed_data/weather_data/weather_data_2016-01-"+str(fileday)

	if os.path.exists(action_file):
		e=pd.read_table(action_file,header=None)
		e=pd.DataFrame(e.get_values(),columns=['Time','Weather','temperature','PM2.5','time_id',
					   'weather_1','weather_2','weather_3','weather_4','weather_5',
					   'weather_6','weather_7','weather_8','weather_9'])
		e.Time=processing_day
		e=e.rename(columns={'Time':'Day'})
		#in 10 min
		temp=e
		temp.time_id=temp.time_id+1
		for i in ['Weather','temperature','PM2.5','weather_1','weather_2','weather_3','weather_4','weather_5','weather_6','weather_7','weather_8','weather_9']:
			temp=temp.rename(columns={i:'In10min_'+i})
		mother_table=pd.merge(mother_table,temp,on=['Day','time_id'],how='left')
		#in 2h10 min
		temp=e
		temp.time_id=temp.time_id+13
		for i in ['Weather','temperature','PM2.5','weather_1','weather_2','weather_3','weather_4','weather_5','weather_6','weather_7','weather_8','weather_9']:
			temp=temp.rename(columns={i:'In2hour_'+i})
		mother_table=pd.merge(mother_table,temp,on=['Day','time_id'],how='left')


	#two_day_ago
	fileday=processing_day-2
	if fileday<10:
		action_file=base_path + "processed_data/weather_data/weather_data_2016-01-0"+str(fileday)
	else:
		action_file=base_path + "processed_data/weather_data/weather_data_2016-01-"+str(fileday)

	if os.path.exists(action_file):
		e=pd.read_table(action_file,header=None)
		e=pd.DataFrame(e.get_values(),columns=['Time','Weather','temperature','PM2.5','time_id',
					   'weather_1','weather_2','weather_3','weather_4','weather_5',
					   'weather_6','weather_7','weather_8','weather_9'])
		e.Time=processing_day
		e=e.rename(columns={'Time':'Day'})
		#in 10 min
		temp=e
		temp.time_id=temp.time_id+1
		for i in ['Weather','temperature','PM2.5','weather_1','weather_2','weather_3','weather_4','weather_5','weather_6','weather_7','weather_8','weather_9']:
			temp=temp.rename(columns={i:'twodayago_In10min_'+i})
		mother_table=pd.merge(mother_table,temp,on=['Day','time_id'],how='left')
		#in 2h10 min
		temp=e
		temp.time_id=temp.time_id+13
		for i in ['Weather','temperature','PM2.5','weather_1','weather_2','weather_3','weather_4','weather_5','weather_6','weather_7','weather_8','weather_9']:
			temp=temp.rename(columns={i:'twodayago_In2hour_'+i})
		mother_table=pd.merge(mother_table,temp,on=['Day','time_id'],how='left')


	#last_week_day
	fileday=processing_day-7
	if fileday<10:
		action_file=base_path + "processed_data/weather_data/weather_data_2016-01-0"+str(fileday)
	else:
		action_file=base_path + "processed_data/weather_data/weather_data_2016-01-"+str(fileday)

	if os.path.exists(action_file):
		e=pd.read_table(action_file,header=None)
		e=pd.DataFrame(e.get_values(),columns=['Time','Weather','temperature','PM2.5','time_id',
					   'weather_1','weather_2','weather_3','weather_4','weather_5',
					   'weather_6','weather_7','weather_8','weather_9'])
		e.Time=processing_day
		e=e.rename(columns={'Time':'Day'})
		#in 10 min
		temp=e
		temp.time_id=temp.time_id+1
		for i in ['Weather','temperature','PM2.5','weather_1','weather_2','weather_3','weather_4','weather_5','weather_6','weather_7','weather_8','weather_9']:
			temp=temp.rename(columns={i:'Weekday_In10min_'+i})
		mother_table=pd.merge(mother_table,temp,on=['Day','time_id'],how='left')
		#in 2h10 min
		temp=e
		temp.time_id=temp.time_id+13
		for i in ['Weather','temperature','PM2.5','weather_1','weather_2','weather_3','weather_4','weather_5','weather_6','weather_7','weather_8','weather_9']:
			temp=temp.rename(columns={i:'Weekday_In2hour_'+i})
		mother_table=pd.merge(mother_table,temp,on=['Day','time_id'],how='left')



	if not os.path.exists(base_path + "processed_data/train_data/"):
		os.makedirs(base_path + "processed_data/train_data/")
	mother_table.to_csv(base_path + "processed_data/train_data/train_day_"+str(processing_day)+'.csv',index=False)
