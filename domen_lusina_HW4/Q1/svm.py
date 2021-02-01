# Python 3.5.2
import time
import matplotlib.pyplot as plt
import numpy as np

C = 100
d = 122  # dimension of w / number of features
n = 6414  # number of samples in the training data


# batch_gradient_descent
def BGD(x, y):
    start = time.time()
    eta = 0.0000003
    epsilon = 0.25

    b = 0
    w = np.zeros([d, 1])
    current_error = epsilon + 1  # we set it so for firs iteration it is true
    # we could use loop over all i in range(n) and add 1 -  y[i] * (np.dot(w, x[i]) + b) to calculate error however since w and b are 0 there is no need
    err = n
    f = [0.5 * sum(w ** 2)[0] + C * err]
    k = 0
    while current_error > epsilon:
        w_old = np.copy(w)

        # updating w
        M = np.multiply(y, np.squeeze(np.dot(x, w_old) + b))
        for j in range(d):
            cost = np.multiply(y, x[:, j])
            grad = 0
            for i, el in enumerate(M):
                if el < 1:
                    grad -= cost[i]
            grad *= C
            w[j] = w[j] - eta * (w_old[j] + grad)

        # updating b
        grad_b = 0
        for i in range(n):
            if M[i] < 1:
                grad_b -= y[i]
        grad_b *= C
        b = b - eta * grad_b

        # we compute the error
        err = 0
        for i in range(n):
            err += max(0, 1 - y[i] * (np.dot(x[i], w) + b))
        f.append(0.5 * sum(w ** 2)[0] + C * err[0])
        current_error = abs((f[k] - f[k + 1]) / f[k] * 100)
        k += 1
    end = time.time()
    total_time = end - start
    return total_time, f


# stackoverflow solution fow random swap
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def SGD(x, y):
    start = time.time()
    eta = 0.0001
    epsilon = 0.001
    x, y = unison_shuffled_copies(x, y)
    b = 0
    w = np.zeros([d, 1])
    current_error = epsilon + 1  # we set it so for firs iteration it is true
    # we could use loop over all i in range(n) and add 1 -  y[i] * (np.dot(w, x[i]) + b) to calculate error however since w and b are 0 there is no need
    err = n
    f = [0.5 * sum(w ** 2)[0] + C * err]
    err_list = [0.0]
    k = 0
    i = 0
    while current_error > epsilon:
        w_old = np.copy(w)
        # updating w
        M = y[i] * np.squeeze(np.dot(x[i], w_old) + b)
        for j in range(d):
            grad = 0
            if M < 1:
                grad -= y[i] * x[i, j]
            w[j] = w[j] - eta * (w_old[j] + C * grad)

        # updating b
        grad_b = 0
        if M < 1:
            grad_b -= y[i]
        b = b - eta * grad_b * C

        # we compute the error
        err = 0
        for h in range(n):
            err += max(0, 1 - y[h] * (np.dot(x[h], w) + b))
        f.append(0.5 * sum(w ** 2)[0] + C * err[0])

        current_error = 0.5 * abs((f[k] - f[k + 1]) / f[k] * 100) + 0.5 * err_list[k]
        err_list.append(current_error)
        k = k + 1
        i = (i % n) + 1

    end = time.time()
    total_time = end - start
    return total_time, f


def MGD(x, y):
    start = time.time()
    eta = 0.00001
    epsilon = 0.01
    x, y = unison_shuffled_copies(x, y)
    b = 0
    batch_size = 20
    w = np.zeros([d, 1])
    current_error = epsilon + 1  # we set it so for firs iteration it is true
    # we could use loop over all i in range(n) and add 1 -  y[i] * (np.dot(w, x[i]) + b) to calculate error however since w and b are 0 there is no need
    err = n
    f = [0.5 * sum(w ** 2)[0] + C * err]
    err_list = [0.0]
    k = 0
    l = 0

    while current_error > epsilon:
        w_old = np.copy(w)
        # updating w
        M = np.multiply(y, np.squeeze(np.dot(x, w_old) + b))
        for j in range(d):
            cost = np.multiply(y, x[:, j])
            grad = 0
            for i in range(int(l * batch_size + 1), int(min(batch_size * (l + 1), n))):
                if M[i] < 1:
                    grad -= cost[i]
            grad *= C
            w[j] = w[j] - eta * (w_old[j] + grad)

        # updating b
        grad_b = 0
        for i in range(int(l * batch_size + 1), int(min(batch_size * (l + 1), n))):
            if M[i] < 1:
                grad_b -= y[i]
        grad_b *= C
        b = b - eta * grad_b

        # we compute the error
        err = 0
        for h in range(n):
            err += max(0, 1 - y[h] * (np.dot(x[h], w) + b))
        f.append(0.5 * sum(w ** 2)[0] + C * err[0])

        current_error = 0.5 * abs((f[k] - f[k + 1]) / f[k] * 100) + 0.5 * err_list[k]
        err_list.append(current_error)
        # print(k, current_error, f[k])
        k = k + 1
        l = (l + 1) % ((n + batch_size - 1) / batch_size)

    end = time.time()
    total_time = end - start
    return total_time, f


features = open('features.txt', 'r')
x = np.array([list(map(float, line.split(','))) for line in features])

y = np.zeros(n)
target = open('target.txt', 'r')
y = np.array([float(x) for x in target])

tim_BGD, f_BGD = BGD(x, y)
tim_SGD, f_SGD = SGD(x, y)
tim_MGD, f_MGD = MGD(x, y)

print("Time batch gradient descend: " + str(tim_BGD) + " s")
print("Time stochastic  gradient descend: " + str(tim_SGD) + " s")
print("Time mini batch  gradient descend: " + str(tim_MGD) + " s")

lines = [f_BGD, f_SGD, f_MGD]
colors = ['r', 'g', 'b']
labels = ['BGD', 'SGD', 'MBGD']

# fig1 = plt.figure()
for i, c, l in zip(lines, colors, labels):
    plt.plot(list(range(len(i))), i, c, label='l')
    plt.legend(labels)
plt.show()
