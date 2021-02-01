from pyspark import SparkConf, SparkContext
import numpy as np

# Python 2.7.12
k = 40


def multiply(rdd, r, transpose=False):
    r_new = np.zeros(r.shape)
    if transpose:
        mul = rdd.map(lambda ((n1, n2), c): (n1, c * r[0, int(n2) - 1])).sortByKey(ascending=True).reduceByKey(
            lambda c1, c2: c1 + c2).collect()
    else:
        mul = rdd.map(lambda ((n2, n1), c): (n1, c * r[0, int(n2) - 1])).sortByKey(ascending=True).reduceByKey(
            lambda c1, c2: c1 + c2).collect()
    for n, c in mul:
        r_new[0, int(n) - 1] += c
    return r_new


conf = SparkConf()
sc = SparkContext(conf=conf)

test = False
if test:
    n = 100
    lines = sc.textFile('graph-small.txt')
else:
    n = 1000
    lines = sc.textFile('graph-full.txt')

data = lines.map(lambda line: tuple(line.split('\t')))
M = data.map(lambda node: (node, 1))  # we create sparse matrix

M = M.reduceByKey(min)  # we remove duplicates

h = np.ones((1, n))

for _ in range(k):
    a = multiply(M, h, False)
    a = a / a.max()
    h = multiply(M, a, True)
    h = h / h.max()


max_nodes_h = h[0].argsort()[-5:][::-1]
min_nodes_h = h[0].argsort()[:5][::1]

print('List of top 5 hubbiness nodes ids with highest hubbiness scores:')
for i, x in enumerate(max_nodes_h):
    print(str(i + 1) + ". node, id = " + str(x + 1) + " and hubbiness score = " + str(h[0, x]))
print('')
print('List of bottom 5 hubbiness nodes ids with highest hubbiness scores:')
for i, x in enumerate(min_nodes_h):
    print(str(i + 1) + ". node, id = " + str(x + 1) + " and hubbiness score = " + str(h[0, x]))


max_nodes_a = a[0].argsort()[-5:][::-1]
min_nodes_a = a[0].argsort()[:5][::1]
print('')
print('')
print('List of top 5 authority nodes ids with highest authority scores:')
for i, x in enumerate(max_nodes_a):
    print(str(i + 1) + ". node, id = " + str(x + 1) + " and authority score = " + str(a[0, x]))
print('')
print('List of bottom 5 authority nodes ids with highest authority scores:')
for i, x in enumerate(min_nodes_a):
    print(str(i + 1) + ". node, id = " + str(x + 1) + " and authority score = " + str(a[0, x]))
sc.stop()
""""""
