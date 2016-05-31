import numpy as np
import matplotlib.pyplot as plt
import pandas
import pylab
import os


for day in range(8, 22):
    table = pandas.read_csv('processed_data/train_data/train_day_{0}.csv'.format(day))
    districts = []
    for _hash in table.start_district_hash:
        if _hash not in districts:
            districts.append(_hash)

    for _hash in districts:
        t = table[[_hash == tmp for tmp in table.start_district_hash]]
        x_axis = t.time_id
        y_axis = t.Order_cnt
        p1 = pylab.plot(x_axis, y_axis, 'g', label='Order_cnt')

        y_axis = t.Gap_cnt
        p2 = pylab.plot(x_axis, y_axis, 'r', label='Gap_cnt')

        pylab.xlabel("time id")
        pylab.ylabel("order number")
        pylab.title("district {0}, day {1} order info".format(t.district_id[0], day))
        pylab.legend()
        if not os.path.exists('./plot'):
            os.makedirs("./plot")
        pylab.savefig('./plot/district_{0}_day_{1}.jpg'.format(t.district_id[0], day))
        pylab.close()
