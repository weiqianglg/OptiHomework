# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
m, n = 30, 100
alpha, beta = 0.3, 0.5
gradient_stop_criteria = 0.001
s, mu = 1, 20
inner_stop_criteria = 0.001

prng = np.random.RandomState(123456789)
A = prng.rand(m, n)
x0 = prng.rand(n)
b = A.dot(x0) + 1
P = np.eye(n)


def f(s, x):
    return 0.5 * s * x.T.dot(P).dot(x) - np.sum(np.log(b - A.dot(x)))


def qp(x):
    return 0.5 * x.T.dot(P).dot(x)


def gradient1(s, x):
    return s * P.dot(x) + A.T.dot(1.0 / (b - A.dot(x)))


def gradient2(s, x):
    return s * P + A.T.dot(np.diag(1.0 / (b - A.dot(x)) ** 2)).dot(A)


def find_doom_t(x, delta_x):
    t = 1.0
    while np.min(b - A.dot(x + t * delta_x)) <= 0:
        t *= beta
    return t


def backtrack_line_t(s, x, delta_x, g1):
    t = find_doom_t(x, delta_x)
    while f(s, x + t * delta_x) > f(s, x) + alpha * t * (np.transpose(g1).dot(delta_x)):
        t *= beta
    return t


def newton_descent_method():
    r_f, r_t = [], []
    x = x0
    for i in range(0, 1000):
        global s
        g1, g2 = gradient1(s, x), gradient2(s, x)
        delta_x = -np.linalg.inv(g2).dot(g1)
        t = backtrack_line_t(s, x, delta_x, g1)
        x = x + t * delta_x
        r_f.append(float(m) / s)
        print s, i, np.transpose(g1).dot(-delta_x), t
        if np.transpose(g1).dot(-delta_x) < 2 * gradient_stop_criteria:
            if float(m) / s < inner_stop_criteria:
                break
            else:
                s *= mu
        r_t.append(t)
    print "alpha {} beta {} iteration {}".format(alpha, beta, i)
    return i, r_f, r_t


def plot(f, t):
    pf = plt.subplot(211)
    pf.plot(range(len(f)), f, '-s')
    pf.set_title("gap vs. iteration")
    pf.set_ylabel("gap")
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