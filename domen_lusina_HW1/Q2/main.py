import csv
import itertools
import operator
from collections import Counter


def read(file):
    f = open(file, "rt", encoding="utf8")
    reader = csv.reader(f, delimiter=" ")
    return [sorted(set(d[:-1])) for d in reader]


data = read("browsing.txt")

T = 100
# we count each item
counter = dict(Counter(list(itertools.chain(*data))))
# we filter out items bellow threshold
fitems = {k: v for k, v in counter.items() if v >= T}
frequent_items = list(fitems.keys())

# SECTION D - generating pairs
pair_keys = list(itertools.combinations(frequent_items, 2))
pairs = {tuple(sorted(pair)): 0 for pair in pair_keys}

for i, basket in enumerate(data):
    comb_basket = [item for item in basket if
                   item in frequent_items]  # we look at each item and find out if it is frequent
    comb_basket = list(itertools.combinations(comb_basket, 2))  # we create pairs from frequent items in basket
    comb_basket = [tuple((sorted(el))) for el in comb_basket]
    for pair in comb_basket:
        pairs[pair] = pairs[pair] + 1
# we check if each pair is above threshold
fpairs = {tuple(sorted(k)): v for k, v in pairs.items() if v >= T}

confidencePairs = {}
for pair, value in fpairs.items():
    confidencePairs[tuple((pair[0], pair[1]))] = value / fitems[pair[0]]
    confidencePairs[tuple((pair[1], pair[0]))] = value / fitems[pair[1]]

# SECTION E - generating triplets
frequent_pairs_keys = list(fpairs.keys())
triplets_keys = []
# we create triplets from pairs -> if pair (a,b) and pair (b,c) is frequent then triplet (a,b,c) could be frequent
for i, t1 in enumerate(frequent_pairs_keys):
    for j in range(i + 1, len(frequent_pairs_keys)):
        t2 = frequent_pairs_keys[j]
        triple = set(t1).union(set(t2))
        if len(triple) == 3:
            triplets_keys.append(tuple(triple))
triplets_keys = set(triplets_keys)
triplets = {triple: 0 for triple in triplets_keys}

for i, basket in enumerate(data):
    comb_basket = list(itertools.combinations(basket, 3))  # we create triplets from items in basket
    comb_basket = [tuple((set(el))) for el in comb_basket]
    for triple in comb_basket:  # we add triplet if triple is a candidate for frequent triple
        if triple in triplets_keys:
            triplets[triple] = triplets[triple] + 1

# filtering out triplets bellow threshold
ftriplets = {k: v for k, v in triplets.items() if v >= T}
confidenceTriplets = {}
for triple, value in ftriplets.items():
    # x y z
    confidenceTriplets[tuple((triple[0], triple[1], triple[2]))] = value / fpairs[tuple(sorted((triple[0], triple[1])))]
    # x z y
    confidenceTriplets[tuple((triple[0], triple[2], triple[1]))] = value / fpairs[tuple(sorted((triple[0], triple[2])))]
    # y z x
    confidenceTriplets[tuple((triple[1], triple[2], triple[0]))] = value / fpairs[tuple(sorted((triple[1], triple[2])))]

# results for d
print(sorted(confidencePairs.items(), reverse=True, key=operator.itemgetter(1))[:5])
# result for e
print(sorted(confidenceTriplets.items(), key=lambda x: (-x[1], x[0]))[:15])

