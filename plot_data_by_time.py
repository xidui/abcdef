import numpy as np

import pandas
import pylab
import os
import matplotlib  
import matplotlib.pyplot as plt
 

for day in range(8, 22):
	table = pandas.read_csv('processed_data/train_data/train_day_{0}.csv'.format(day)).fillna(0)

	  
	# with legend  
	plt.figure(day)  
	t = table[[i in range(42,52) for i in table.time_id ] ].reset_index()
	p1 = plt.scatter(t.Gap_cnt, t.In30Min_Gap_cnt, marker = 'x', color = 'm', label='1', s = 100)  

	t = table[ [i in range(65,85) for i in table.time_id ]  ].reset_index()
	p2 = plt.scatter(t.Gap_cnt, t.In30Min_Gap_cnt, marker = '+', color = 'r', label='2', s = 100)  

	t = table[ [i in range(97,107) for i in table.time_id ]  ].reset_index()
	p3 = plt.scatter(t.Gap_cnt, t.In30Min_Gap_cnt, marker = 'o', color = 'c', label='3', s = 100)  
		
	#t = table[ [i in range(122,132) for i in table.time_id ]  ].reset_index()
	#p4 = plt.scatter(t.Gap_cnt, t.In30Min_Gap_cnt, marker = '*', color = 'g', label='4', s = 100)  
	plt.legend(loc = 'upper right')  
	plt.savefig("./plot/by_time_day_"+str(day)+".png")
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
