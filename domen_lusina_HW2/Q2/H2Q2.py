from pyspark import SparkConf, SparkContext
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def l1(a, b):
    return sum(abs(a - b))


def l2(a, b):
    return sum(pow(a - b, 2))


def closest_center(p, centers, l):
    """
    dist = centers.map(lambda el: (l(p, el[0]), el[1])).min()[1]
    print(dist.min()[1])
    return dist.min()[1]
    """
    index = 0
    closest = float("+inf")
    for i, center in enumerate(centers):
        tmp_dist = l(p, center)
        if tmp_dist < closest:
            closest = tmp_dist
            index = i
    return index, (p, 1, closest)


def kmeans(l, centers, data, k=10, MAX_ITER=20):
    costs = []
    noPoints = data.count()
    for iteration in range(MAX_ITER + 1):
        # print("Iteration " + str(iteration))
        clusters = data.map(lambda point: (closest_center(point, centers, l)))
        # print([x[1][2] for x in clusters.collect()])

        cost = clusters.map(lambda x: x[1][2]).reduce(lambda x, y: x + y)
        costs.append(cost / noPoints)

        center_scores = clusters.reduceByKey(lambda n1, n2: (n1[0] + n2[0], n1[1] + n2[1]))

        new_center = center_scores.map(lambda el: (el[0], el[1][0] / el[1][1])).collect()

        no_same_elements = 0
        for (i, center) in new_center:
            if (centers[i] == center).all():
                no_same_elements += 1
            centers[i] = center
        if no_same_elements == k:
            print("Iteration " + str(iteration))
            break
    return costs


def plotError(costs1, costs2, title="", xtitle="", ytitle=""):
    plt.figure()
    plt.plot(list(range(len(costs1))), costs1, 'b', label="C1")
    plt.plot(list(range(len(costs2))), costs2, 'r', label="C2")
    red_patch = mpatches.Patch(color='blue', label='C1')
    blue_path = mpatches.Patch(color='red', label='C2')
    plt.legend(handles=[red_patch, blue_path])
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)
    plt.show()


conf = SparkConf().setMaster("local").setAppName("kMeans")
sc = SparkContext(conf=conf)
lines = sc.textFile("data.txt")
data = lines.map(lambda line: np.matrix(line).A1).cache()

c1 = sc.textFile("c1.txt")
centers1 = c1.map(lambda line: np.matrix(line).A1).collect()

c2 = sc.textFile("c2.txt")
centers2 = c2.map(lambda line: np.matrix(line).A1).collect()

# Euclidean distance
print("Euclidean distance")
print("C1")
costsC1 = kmeans(l2, centers1, data)
print("C1 % decrease after 10 iterations:")
print(1 - costsC1[10] / costsC1[0])

print("C2")
costsC2 = kmeans(l2, centers2, data)
print("C2 % decrease after 10 iterations:")
print(1 - costsC2[10] / costsC2[0])
plotError(costsC1, costsC2, "L2 ", "iterations", "cost")

#########################################################
# Manhattan distance
lines = sc.textFile("data.txt")
data = lines.map(lambda line: np.matrix(line).A1).cache()

c1 = sc.textFile("c1.txt")
centers1 = c1.map(lambda line: np.matrix(line).A1).collect()

c2 = sc.textFile("c2.txt")
centers2 = c2.map(lambda line: np.matrix(line).A1).collect()

print()
print("Manhattan distance")
print("C1")
costsC1 = kmeans(l1, centers1, data)
print("C1 % decrease after 10 iterations:")
print(1 - costsC1[10] / costsC1[0])

print("C2")
costsC2 = kmeans(l1, centers2, data)
print("C2 % decrease after 10 iterations:")
print(1 - costsC2[10] / costsC2[0])

plotError(costsC1, costsC2, "L2 ", "iterations", "cost")

sc.stop()
