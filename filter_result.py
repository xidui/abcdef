# f = open('./test_set_1/read_me_1.txt')
# d = {}
# line = f.readline().strip()
# while line:
#     d[line] = 1
#     line = f.readline().strip()
# f.close()
#
# f = open('./result.csv')
# target = open('./result2.csv', 'w')
#
# line = f.readline()
# while line:
#     time = line.split(',')[1]
#     if time in d:
#         target.write(line)
#     line = f.readline()
# f.close()
# target.close()

f = open('./result2.csv')
target = open('./result3.csv', 'w')

line = f.readline().strip()
while line:
    l = line.split(',')
    l[2] = '1\n'

    target.write(','.join(l))
    line = f.readline().strip()


f.close()
