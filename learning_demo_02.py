#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat

# y = 2x
X = mat([1, 2, 3]).reshape(3, 1)
Y = 2 * X

# 最小二乘法
# theta = (X'X)^-1X'Y
theta = dot(dot(inv(dot(X.T, X)), X.T), Y)

# 梯度下降法
# theta = theta - alpha*(theta*X-Y)*X
theta = 1.
alpha = 0.1
for i in range(100):
    theta = theta + alpha * np.sum((Y - dot(X, theta))
                                   * X.reshape(1, 3)) / len(X)

# np.sum 对矩阵加权平均
print(theta)
