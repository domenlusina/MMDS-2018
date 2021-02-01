from pyspark import SparkConf, SparkContext
import numpy as np
# Python 2.7.12
beta = 0.8
k = 40


def multiply(rdd, r):
    r_new = np.zeros(r.shape)
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

# count edges that point from each node
edges = M.map(lambda ((n1, n2), _): (n1, 1))
counted_edges = edges.reduceByKey(lambda n1, n2: n1 + n2)
counted_edges = counted_edges.collectAsMap()

# matrix M normalization + beta multiplication
M_norm = M.map(lambda ((n1, n2), c1): ((n1, n2), beta * c1 / counted_edges[n1]))

r = np.full((1, n), 1.0 / n)
teleport = (1 - beta) / n

for _ in range(k):
    r1 = np.full((1, n), teleport)
    r2 = multiply(M_norm, r)
    r = r1 + r2

max_nodes_i = r[0].argsort()[-5:][::-1]
min_nodes_i = r[0].argsort()[:5][::1]

print('List of top 5 nodes ids with highest PageRank scores:')
for i, x in enumerate(max_nodes_i):
    print(str(i + 1) + ". node, id = " + str(x + 1) + " and PageRank score = " + str(r[0, x]))
print('')
print('List of bottom 5 nodes ids with highest PageRank scores:')
for i, x in enumerate(min_nodes_i):
    print(str(i + 1) + ". node, id = " + str(x + 1) + " and PageRank score = " + str(r[0, x]))

sc.stop()
