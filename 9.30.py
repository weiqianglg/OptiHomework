import numpy as np
import matplotlib.pyplot as plt
m, n = 100, 500
alpha, beta = 0.3, 0.5
gradient_stop_criteria = 0.001

prng = np.random.RandomState(123456789)
A = prng.normal(0, 1, size=(m, n))


def f(x):
    return -np.sum(np.log(1 - np.dot(A, x))) - np.sum(np.log(1 - np.power(x, 2)))


def gradient1(x):
    return np.dot(np.transpose(A), 1/(1 - np.dot(A, x))) + 2*x/(1 - np.power(x, 2))


def gradient2(x):
    return np.transpose(A).dot(np.diag(1/(1 - np.dot(A, x))**2)).dot(A) + np.diag(2*(1+x**2)/(1-x**2)**2)


def find_doom_t(x, delta_x):
    t = 1
    while max(np.dot(A, x + t * delta_x)) >= 1 or max(np.abs(x + t * delta_x)) >= 1:
        t *= beta
    return t


def backtrace_line_t(x, delta_x, g1):
    t = find_doom_t(x, delta_x)
    while f(x + t * delta_x) > f(x) + alpha * t * (np.transpose(g1).dot(delta_x)):
        t *= beta
    return t


def gradient_descent_method():
    r_f, r_t = [], []
    x = np.zeros(n)
    for i in range(0, 1000):
        g = gradient1(x)
        delta_x = -g
        if np.linalg.norm(g) < gradient_stop_criteria:
            break
        t = backtrace_line_t(x, delta_x, g)
        x = x + t * delta_x
        r_f.append(f(x))
        r_t.append(t)
    print "alpha {} beta {} iteration {}".format(alpha, beta, i)
    return i, r_f, r_t


def newton_descent_method():
    r_f, r_t = [], []
    x = np.zeros(n)
    for i in range(0, 1000):
        g1, g2 = gradient1(x), gradient2(x)
        delta_x = -np.linalg.inv(g2).dot(g1)
        if np.transpose(g1).dot(-delta_x) < 2 * gradient_stop_criteria:
            break
        t = backtrace_line_t(x, delta_x, g1)
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


def adjust_beta():
    ys = np.linspace(0, 0.97)
    ri = []
    for y in ys:
        global beta
        beta = y
        iteration, _, __ = optimize_func()
        ri.append(iteration)
    plot2(ys, ri, "beta")


def adjust_alpha():
    ys = np.linspace(0, 0.5)
    ri = []
    for y in ys:
        global alpha
        alpha = y
        iteration, _, __ = optimize_func()
        ri.append(iteration)
    plot2(ys, ri, "alpha")

optimize_func = newton_descent_method


if __name__ == "__main__":
    iteration, f, t = optimize_func()
    plot(f, t)
    # adjust_alpha()
    # adjust_beta()
    print "done."

