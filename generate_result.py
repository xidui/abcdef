# -*- coding: utf-8 -*-
import math
import pandas as pd
import os


import warnings
warnings.filterwarnings("ignore")


a = pd.DataFrame(columns=['distinct_id', 'time_slot', 'prediction'])
for filename in os.listdir("./result"):
    a = a.append(pd.read_csv("./result/"+filename)).fillna(0)

a = a.groupby(['distinct_id', 'time_slot']).sum().reset_index()
a.prediction = [round(i, 2) for i in a.prediction]
a.distinct_id = [int(i) for i in a.distinct_id ]
a.to_csv('result.csv', index=False, header=False)
print "finished " + str(len(a))