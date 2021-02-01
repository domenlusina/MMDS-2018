import re
import sys
from pyspark import SparkConf, SparkContext
import heapq
from operator import itemgetter


def line2friends(line):
    # for the given input in form <User><TAB><Friends>
    # e.g. "Elise    John,Matt"
    # we output an array of pairs which represent "friendship"
    # for the given example the output would be [("Elise","John"),("Elise","Matt")]
    user, friends = line.split("\t")
    if len(friends) > 0:
        friends = friends.split(",")
        return [(int(user), int(friend)) for friend in friends]
    else:
        # in case user has no friends
        return []


conf = SparkConf().setMaster("local[*]")
sc = SparkContext(conf=conf)
lines = sc.textFile("/HW1/soc-LiveJournal1Adj.txt")
# we transform each line into tuple (u1,u2) that represents friendship between u1 and u2
friendships = lines.flatMap(lambda line: line2friends(line))
print("friendships")
# we preform a self join that output line in a form (U,(T,V)), we filter out the values (T,V) and remove lines
# where T==V you can't be friend with yourself?
joined = friendships.join(friendships).values().filter(lambda pair: pair[0] != pair[1])
print("joined")
# we remove people we are already friends with and preform reduce
# subtract would be a better method but for some reason my process got stuck and didn't finish

# BUG ?
# new_friendship = joined.subtract(friendships)
# console outputs [Stage 4:=======================================>                   (2 + 0) / 3]
# and doesn't proceed even when using very small data
friendships_collected = friendships.collect()
new_friends = joined.filter(lambda pair: pair not in friendships_collected).map(lambda pair: (pair, 1)).reduceByKey(
    lambda n1, n2: n1 + n2)
print("new_friends")
# we change the format from ((u1, u2), count) (where u1 is user1 and u2 is user2 and count in number of mutual friends)
# to (u1,(u2,count)) and preform grouping by key (all u1 are placed together)
# we now have a row in form (u1, [(u2,count2),(u3, count3),...]) where u2,u3,... are friend recommendation with count2,count3 mutual friends
# then we preform map function so we transform iterable into list and sort by u2,u3 values in ascending order
new_friends_grouped = new_friends \
    .map(lambda ((u1, u2), count): (u1, (u2, count))) \
    .groupByKey() \
    .map(lambda x: (x[0], sorted(list(x[1]),key=lambda el: el[0]) ))
print("new_friends_grouped")
# finally  inner arrays of mutual friends and length is reduced to 10 as instructed
new_friends_best10 = new_friends_grouped.map(lambda (user, arr): (user,[ x[0] for x in heapq.nlargest(10, arr, key=itemgetter(1))]))
print("new_friends_best10")
# print(friendships.collect())
# print
# print(joined.collect())
#print(new_friends_grouped.collect())
#print
#print(new_friends_best10.collect())
new_friends_best10.saveAsTextFile("/HW1/results")
sc.stop()


