import csv
import math
import numpy as np


def read_data(file):
    f = open(file, "rt", encoding="utf8")
    reader = csv.reader(f, delimiter="\n")
    data = [d[0] for d in reader]
    return data


shows = read_data('shows.txt')
showDictionary = {i: show for i, show in enumerate(shows)}

# alex = np.array(read_data('alex.txt')[0])

R = np.matrix(';'.join(read_data('user-shows.txt')))

P = np.sum(R, axis=1).A1
Q = np.sum(R, axis=0).A1

Q_1 = np.diag(1 / np.power(Q, 0.5))
P_1 = np.diag(1 / np.power(P, 0.5))

S_movie = Q_1 * R.transpose() * R * Q_1
movie_movie = R * S_movie

S_user = P_1 * R * R.transpose() * P_1
user_user = S_user * R

indices_user = user_user[499, :100].A1.argsort()[-5:][::-1]
indices_movie = movie_movie[499, :100].A1.argsort()[-5:][::-1]
print("User-user recommendations:")
print([showDictionary[x] for x in indices_user])
print([user_user[499, x] for x in indices_user])
print("Movie-movie recommendations:")
print([showDictionary[x] for x in indices_movie])
print([movie_movie[499, x] for x in indices_movie])
