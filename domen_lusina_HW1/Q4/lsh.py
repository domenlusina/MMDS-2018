# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla

import operator
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def l1(u, v):
    return sum(abs(u - v))


def load_data(filename):
    # replaced genfromtxt with loadtxt since I ran out of memory otherwise
    return np.loadtxt(filename, delimiter=',')


def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))

    return f


def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low=0,
                                       high=num_dimensions,
                                       size=k)
        thresholds = np.random.randint(low=min_threshold,
                                       high=max_threshold + 1,
                                       size=k)

        functions.append(create_function(dimensions, thresholds))
    return functions


def hash_vector(functions, v):
    return np.array([f(v) for f in functions])


def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))


def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
                            any(hashed_point == hashed_A[i]), range(len(hashed_A)))


def lsh_setup(A, k=24, L=10):
    functions = create_functions(k=k, L=L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)


def lsh_search(A, hashed_A, functions, query_index, num_neighbors=10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)

    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]


def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")


def linear_search(A, query_index, num_neighbors):
    distances = [(0, float('Inf'))] * len(A)
    for i, image in enumerate(A):
        if i != query_index:
            d = l1(image, A[query_index, :])
            distances[i] = tuple((i, d))
    sorted_distances = sorted(distances, key=operator.itemgetter(1))[:num_neighbors]
    return [i for i, d in sorted_distances]


# Function that computes the error measure
def calc_error(A, rows, k=24, L=10, num_neighbors=3):
    functions, hashed_A = lsh_setup(A, k=k, L=L)
    err = 0
    for row in rows:
        top_3_lsh = lsh_search(A, hashed_A, functions, row, num_neighbors)
        top_3_linear = linear_search(A, row, num_neighbors)
        numerator = sum([l1(A[row, :], el) for el in A[top_3_lsh, :]])
        denominator = sum([l1(A[row, :], el) for el in A[top_3_linear, :]])
        err += numerator / denominator
    return err / len(rows)


def problem4():
    A = load_data("patches.csv")

    rows = [100 * (x + 1) - 1 for x in range(10)]  # -1 because instructions were given for MATLAB
    num_neighbors = 3

    # FIRST part of D section

    # we compute search time for LSH
    t_start = time.time()
    functions, hashed_A = lsh_setup(A)
    """
    for i, row in enumerate(rows):
        lsh_search(A, hashed_A, functions, row, num_neighbors)
    print("LSH time: " + str((time.time() - t_start)/len(rows)) + " s")

    # and for linear search
    t_start = time.time()
    for i, row in enumerate(rows):
        linear_search(A, row, num_neighbors)
    print("Linear search time: " + str((time.time() - t_start)/len(rows)) + " s")
    
    # SECOND part of 4.D

    # testing on different L
    print("Testing on different L")
    L = list(range(10, 21, 2))
    K = 24
    err_L = [0] * len(L)
    for i, li in enumerate(L):
        err_L[i] = calc_error(A, rows, K, li)

    # plotting
    print("Plotting 1")
    plt.plot(L, err_L, 'ro')
    plt.ylabel('Error')
    plt.xlabel('L')
    plt.savefig('HW_4d_1.png')
    # plt.show()

    # testing on diffrent K
    print("Testing on different K")
    L = 10
    K = list(range(16, 25, 2))
    err_K = [0] * len(K)
    for i, ki in enumerate(K):
        err_K[i] = calc_error(A, rows, ki, L)

    print("Plotting 2")
    plt.figure()
    plt.plot(K, err_K, 'ro')
    plt.ylabel('Error')
    plt.xlabel('K')
    plt.savefig('HW_4d_2.png')
    # plt.show()
    """
    # LAST part - 10 near neighbours for row 100 (-1)
    row = 99
    num_neighbors = 10
    top_10_lsh = lsh_search(A, hashed_A, functions, row, num_neighbors)
    top_10_linear = linear_search(A, row, num_neighbors)

    plt.subplot(2, 11, 1)
    original_image = (A[row, :]).reshape(20, 20)
    plt.imshow(original_image, cmap='gray')
    for i, el in enumerate(top_10_lsh):
        plt.subplot(2, 11, i + 2)
        image = (A[el, :]).reshape(20, 20)
        plt.imshow(image, cmap='gray')

    plt.subplot(2, 11, num_neighbors + 2)
    original_image = (A[row, :]).reshape(20, 20)
    plt.imshow(original_image, cmap='gray')
    for i, el in enumerate(top_10_linear):
        plt.subplot(2, 11, i + num_neighbors + 3)
        image = (A[el, :]).reshape(20, 20)
        plt.imshow(image, cmap='gray')
    plt.savefig('HW_4d_3.png')
    plt.show()

#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0, :]), 6)
        self.assertEqual(f2(A[0, :]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

        ### Write your tests here (they won't be graded,
        ### but you may find them helpful)


if __name__ == '__main__':
    # unittest.main()
    problem4()
