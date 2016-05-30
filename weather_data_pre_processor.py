import os
from traffic_data_pre_processor import get_timestamp_by_id

def get_time_stamp(t):
    tmp = t.split()[1].split(':')
    if int(tmp[1]) % 10 >= 5:
        return -1
    return int(tmp[0]) * 6 + int(tmp[1]) / 10 + 1


def _fill_data(dic):
    sorted(dic.items(), key=lambda i: i[0])
    tmp = {}
    itr = iter(dic.items())
    ptr1 = itr.next()
    while True:
        try:
            ptr2 = itr.next()
        except StopIteration:
            break
        if ptr2[0] - ptr1[0] == 1:
            ptr1 = ptr2
            continue

        for time in range(ptr1[0] + 1, ptr2[0]):
            data = []
            # append the time stamp
            data.append(ptr1[1][0].split()[0] + ' ' + get_timestamp_by_id(time))

            # append the weather
            # since it is number, we can not use mean or medium, just use the previous one
            data.append(ptr1[1][1])

            # calculate the temperature
            r1, r2 = float(ptr1[1][2]), float(ptr2[1][2])
            target = (r1 * (ptr2[0] - time) + r2 * (time - ptr1[0])) / (ptr2[0] - ptr1[0])
            data.append('{0}'.format(target))

            # calculate the pm2.5
            r1, r2 = float(ptr1[1][3]), float(ptr2[1][3])
            target = (r1 * (ptr2[0] - time) + r2 * (time - ptr1[0])) / (ptr2[0] - ptr1[0])
            data.append('{0}'.format(target))

            tmp[time] = data

        ptr1 = ptr2

    for k, v in tmp.items():
        dic[k] = v
    sorted(dic.items(), key=lambda i: i[0])

    # fill head
    ptr = 0
    while ptr not in dic:
        ptr += 1
    for i in range(0, ptr):
        dic[i] = dic[ptr]

    # fill tail
    ptr = 143
    while ptr not in dic:
        ptr -= 1
    for i in range(ptr + 1, 144):
        dic[i] = dic[ptr]
    sorted(dic.items(), key=lambda i: i[0])


for filename in os.listdir("./training_data/weather_data"):
    if filename == '.DS_Store':
        continue
    f = open('./training_data/weather_data/' + filename)
    line = f.readline()
    dic = {}

    while line:
        words = line.split('\t')
        timestamp = get_time_stamp(words[0])
        if timestamp >= 0:
            dic[timestamp] = words
        line = f.readline()

    _fill_data(dic)

    if not os.path.exists("./processed_data/weather_data/"):
        os.makedirs("./processed_data/weather_data/")
    target_file = open("./processed_data/weather_data/" + filename, 'w')
    for timestamp, value in dic.items():
        matrix = ['0'] * 9
        matrix[int(value[1]) - 1] = '1'
        target_file.write('{0}\t{1}\t{2}\n'.format(
            '\t'.join(value).strip(),
            timestamp,
            '\t'.join(matrix)
        ))
    target_file.close()
    f.close()