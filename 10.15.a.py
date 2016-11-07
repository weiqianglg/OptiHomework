# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
p, n = 30, 100
alpha, beta = 0.3, 0.5
gradient_stop_criteria = 0.0001

prng = np.random.RandomState(123456789)
A = prng.normal(0, 1, size=(p, n))
assert np.linalg.matrix_rank(A) == p
x0 = prng.rand(n)
b = A.dot(x0)


def f(x):
    return np.dot(x, np.log(x))


def gradient1(x):
    return 1.0 + np.log(x)


def gradient2(x):
    return np.diag(1.0 / x)


def find_doom_t(x, delta_x):
    t = 1
    while np.min(x + t * delta_x) <= 0:
        t *= beta
    return t


def backtrack_line_t(x, delta_x, g1):
    t = find_doom_t(x, delta_x)
    while f(x + t * delta_x) > f(x) + alpha * t * (np.transpose(g1).dot(delta_x)):
        t *= beta
    return t


def newton_descent_method():
    r_f, r_t = [], []
    x = x0
    for i in range(0, 1000):
        g1, g2 = gradient1(x), gradient2(x)
        kkt_matrix = np.row_stack((
            np.column_stack((g2, np.transpose(A))),
            np.column_stack((A, np.zeros((p, p))))
            ))
        delta_x_w = np.linalg.inv(kkt_matrix).dot(np.hstack((-g1, np.zeros(p))))
        delta_x = delta_x_w[:n]
        if np.transpose(g1).dot(-delta_x) < 2 * gradient_stop_criteria:
            break
        t = backtrack_line_t(x, delta_x, g1)
        x = x + t * delta_x
        r_f.append(f(x))
        r_t.append(t)
    print "alpha {} beta {} iteration {}".format(alpha, beta, i)
    return i, r_f, r_t


def plot(f, t):
    pf = plt.subplot(211)
    pf.plot(range(len(f)), f, '-s')
    pf.set_title("f(x) vs. iteration")
    pf.set_ylabel("f(x)")
    pt = plt.subplot(212)
    pt.plot(range(len(t)), t, '-d')
    pt.set_title("t(step) vs. iteration")
    pt.set_xlabel("iteration")
    pt.set_ylabel("t")
    plt.show()


def plot2(x, y, key="alpha"):
    pt = plt.subplot(111)
    pt.plot(x, y, '-^')
    pt.set_title("iteration vs. {}".format(key))
    pt.set_ylabel("iteration")
    pt.set_xlabel(key)
    plt.show()


if __name__ == "__main__":
    r_i, r_f, r_t = newton_descent_method()
    plot(r_f, r_t)
    print "done."