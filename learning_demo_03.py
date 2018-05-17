#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat
import pandas as pd

dataset = pd.read_csv('learning_demo_03.csv')

# 取出 X
temp = dataset.iloc[:, 2:5]
temp['x0'] = 1
# 此时 X 内的数据排列为 1,2,3,0
X = temp.iloc[:, [3, 0, 1, 2]]

# 取出 Y
Y = dataset.iloc[:, 1].values.reshape(len(X), 1)

# 最小二乘法
theta = dot(dot(inv(dot(X.T, X)), X.T), Y)
print('theta 1:\n', theta)

# 梯度下降法
theta = np.array([1., 1., 1., 1.]).reshape(4, 1)
alpha = 0.1
temp = theta
X0 = X.iloc[:, 0].values.reshape(len(X), 1)
X1 = X.iloc[:, 1].values.reshape(len(X), 1)
X2 = X.iloc[:, 2].values.reshape(len(X), 1)
X3 = X.iloc[:, 3].values.reshape(len(X), 1)

for i in range(10000):
    theta[0] = theta[0] + alpha * np.sum((Y - dot(X, theta)) * X0) / len(X)
    theta[1] = theta[1] + alpha * np.sum((Y - dot(X, theta)) * X1) / len(X)
    theta[2] = theta[2] + alpha * np.sum((Y - dot(X, theta)) * X2) / len(X)
    theta[3] = theta[3] + alpha * np.sum((Y - dot(X, theta)) * X3) / len(X)
    theta = temp

print('theta 2:\n', theta)
