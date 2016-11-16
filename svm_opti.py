from __future__ import division
import numpy as np
np.random.seed(1)
n = 20
m = 1000
DENSITY = 0.2
beta_true = np.random.randn(n,1)
idxs = np.random.choice(range(n), int((1-DENSITY)*n), replace=False)
for idx in idxs:
    beta_true[idx] = 0
offset = 0
sigma = 45
X = np.random.normal(0, 5, size=(m,n))
Y = np.sign(X.dot(beta_true) + offset + np.random.normal(0,sigma,size=(m,1)))

from cvxpy import *
beta = Variable(n)
v = Variable()
loss = sum_entries(pos(1 - mul_elemwise(Y, X*beta - v)))
reg = norm(beta, 2)**2
lambd = Parameter(sign="positive")
prob = Problem(Minimize(loss/m + lambd*reg))
train_error = []
for i in np.linspace(0,5,50):
    lambd.value = i
    prob.solve()
    train_error.append((np.sign(X.dot(beta_true) + offset) != np.sign(X.dot(beta.value) - v.value)).sum()/m)

import matplotlib.pyplot as plt
plt.plot(np.linspace(0,5,50), train_error)
plt.show()