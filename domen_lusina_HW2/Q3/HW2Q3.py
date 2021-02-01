import copy
import math

import matplotlib.pyplot as plt
import numpy as np


class MatrixFactorizationClass():
    def __init__(self, data_path, k=20):
        self.k = k
        self.data_path = data_path
        self.users = []
        self.movies = []

        with open(self.data_path, "rt", encoding="utf8") as f:
            for row in f:
                row = row.split("\t")
                if row[0] not in self.users:
                    self.append = self.users.append(row[0])
                if row[1] not in self.movies:
                    self.movies.append((row[1]))

        self.P = {}
        self.Q = {}

        for user in self.users:
            self.P[user] = np.random.uniform(0, math.sqrt(5 / self.k), self.k)

        for movie in self.movies:
            self.Q[movie] = np.random.uniform(0, math.sqrt(5 / self.k), self.k)

    def compute_error(self, P, Q, lam):
        S = 0
        with open(self.data_path, "rt", encoding="utf8") as f:
            for row in f:
                user, movie, rui = row.split("\t")
                S += pow(float(rui) - (P[user]).dot(Q[movie]), 2)

        for movie in Q.keys():
            S += lam * Q[movie].dot(Q[movie])
        for user in P.keys():
            S += lam * P[user].dot(P[user])
        return S

    def run(self, lam=0.1, eta=0.03, max_iterations=40):
        print("Running...")
        Errors = []
        Errors.append(self.compute_error(self.P, self.Q, lam))

        for s in range(max_iterations + 1):
            # print("Iteration: " + str(s))
            Q = copy.deepcopy(self.Q)
            P = copy.deepcopy(self.P)

            # print(self.songGrades)
            with open(self.data_path, "rt", encoding="utf8") as f:
                for row in f:
                    user, movie, rui = row.split("\t")
                    rui2 = (P[user]).dot(Q[movie])

                    eui = 2 * (float(rui) - rui2)
                    cf = P[user]
                    mf = Q[movie]

                    for k in range(self.k):
                        P[user][k] += eta * (eui * mf[k] - 2 * lam * cf[k])
                        Q[movie][k] += eta * (eui * cf[k] - 2 * lam * mf[k])

            Errors.append(self.compute_error(P, Q, lam))

            if np.array_equal(self.Q, Q) and np.array_equal(self.P, P):
                break

            self.Q = Q
            self.P = P
        return Errors


mf = MatrixFactorizationClass("ratings.train.txt")
Errors = mf.run()
plt.plot(list(range(len(Errors))), Errors)
plt.ylabel("E")
plt.xlabel("Iterations")
print("Errors:")
print(Errors)
plt.show()
