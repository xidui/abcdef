import os


def get_time_id(timestamp):
    t = timestamp.split()[1].split(':')
    return int(t[0]) * 6 + int(t[1]) / 10 + 1


def get_timestamp_by_id(id):
    return "{0:0>2}:{1:0>2}:00".format(id / 6, (id % 6) * 10)


def fill_data(dic):
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
            v1, v2 = ptr1[1], ptr2[1]
            data = []
            # append the region id
            data.append(ptr1[1][0])

            # append the road info with type 1,2,3,4
            for i in range(4):
                r1, r2 = int(v1[i + 1].split(':')[1]), int(v2[i + 1].split(':')[1])
                target1 = (r1 * (ptr2[0] - time) + r2 * (time - ptr1[0])) / (ptr2[0] - ptr1[0])
                data.append('{0}:{1}'.format(i + 1, int(target1)))

            # append the time info
            data.append(get_timestamp_by_id(time))

            # add the data into tmp
            tmp[time] = data

        ptr1 = ptr2

    for k, v in tmp.items():
        dic[k] = v
    sorted(dic.items(), key=lambda i: i[0])

for filename in os.listdir("./training_data/traffic_data"):
    if filename == '.DS_Store':
        continue
    f = open('./training_data/traffic_data/' + filename)
    line = f.readline()
    dic = {}
    '''
    dic = {
        '1ecbb52d73c522f184a6fc53128b1ea1': {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            ...
            144: [],
        },
        '1ecbb52d73c522f184a6fc53128b1ea1': {

        },
    }
    '''
    while line:
        words = line.split('\t')
        if words[0] not in dic:
            dic[words[0]] = {}
        dic[words[0]][get_time_id(words[5])] = words

        line = f.readline()

    for k, v in dic.items():
        fill_data(v)

    if not os.path.exists("./processed_data/traffic_data/"):
        os.makedirs("./processed_data/traffic_data/")
    target_file = open("./processed_data/traffic_data/" + filename, 'w')
    for road_id, road_data in dic.items():
        for time_segment, value in road_data.items():
            target_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(
                value[0],
                value[1],
                value[2],
                value[3],
                value[4],
                value[5].strip(),
                time_segment
            ))
    target_file.close()
    f.close()