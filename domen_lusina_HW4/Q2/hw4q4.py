# Python 3.5.2
import sys

import matplotlib.pyplot as plt
import numpy as np


def hash_fun(a, b, p, n_buckets, x):
    y = x % p
    hash_val = (a * y + b) % p

    return hash_val % n_buckets


testing = False
p = 123457  # prime
delta = np.e ** -5
epsilon = np.e * (10 ** -4)
n_buckets = int(np.e / epsilon)  # buckets

# reading hash param
hash_params = open('hash_params.txt', 'r')
a = []
b = []
for line in hash_params:
    x, y = line.split('\t')
    a.append(int(x))
    b.append(int(y))

a = np.array(a)
b = np.array(b)

hash_matrix = np.zeros((a.size, n_buckets))

if testing:
    words_stream = open('words_stream_tiny.txt', 'r')
    counts = open('counts_tiny.txt', 'r')
else:
    words_stream = open('words_stream.txt', 'r')
    counts = open('counts.txt', 'r')

for i, line in enumerate(words_stream):
    word = int(line.strip())
    h_val = hash_fun(a, b, p, n_buckets, word)
    indices = [list(range(a.size)), h_val.tolist()]
    hash_matrix[indices] += 1


# computing error
n_records = i + 1
err_list = []
freq_list = []
for i, line in enumerate(counts):
    j, fj = map(int, line.split('\t'))
    h_val = hash_fun(a, b, p, n_buckets, j)

    fj_tilde = sys.maxsize
    for k in range(a.size):
        fj_tilde = min(fj_tilde, hash_matrix[k, h_val[k]])
    err = (fj_tilde - fj) / fj
    err_list.append(err)
    freq_list.append(fj / n_records)

plt.loglog(freq_list, err_list, ".")
plt.ylabel("relative error ")
plt.xlabel("word freq. ")
plt.show()
