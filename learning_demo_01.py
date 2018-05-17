#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from numpy import mat
from numpy import dot

A = np.mat([1, 1])
print('A:\n', A)

B = mat([[1, 2], [2, 3]])
print('B:\n', B)

# A(1*2) B(2*2)
print('A.B:\n', dot(A, B))

# 转置
print('A.T:\n', A.T)

# 逆
print('B的逆\n', inv(B))

# 重组新矩阵
print('A reshape:\n', A.reshape(2, 1))
