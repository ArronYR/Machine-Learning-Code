#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """
    ata: 学习速率
    n_iter: 权重向量训练的次数
    w_: 神经分叉权重向量
    errors_: 用于记录神经元判断出错的次数
    """

    def __init__(self, eta=.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        pass

    def net_input(self, X):
        """
        点积运算 W.X
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass

    def predict(self, X):
        """
        预测函数
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        pass

    def fit(self, X, y):
        """
        输入训练数据，培训神经元
        X: 输入样本向量
        y: 对应的样本分类
        x.shape[x_samples, x_features]
        x = [[1, 2, 3], [4, 5, 6]]
        x_samples = x.shape[1] = 2
        x_features = x.shape[2] = 3

        初始化权重为0，第一个为步调函数阈值权重w0
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # update = v * (y - y')
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass
            pass


def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    输入测试数据，画出区域
    """
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 构造验证数据
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    # 在两个区域之间划分界限
    plt.contourf(xx1, xx2, z, alpha=.4, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    for idx, vl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == vl, 0], y=X[y == vl, 1], alpha=0.8,
                    c=cmap(idx), marker=markers[idx], label=vl)
    plt.xlabel('length of the huajing')
    plt.ylabel('length of the huaban')
    plt.legend(loc='upper right')
    plt.show()


def main():
    """
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = [1, -1]
    """
    df = pd.read_csv('learning_demo_05.csv', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 1, -1)
    X = df.iloc[0:100, [0, 2]].values

    # 绘制
    # plt.scatter(X[:50, 0], X[0:50, 1], color='red', marker='o', label='setosa')
    # plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
    #             marker='x', label='versicolor')
    # plt.xlabel('length of the huajing')
    # plt.ylabel('length of the huaban')
    # plt.legend(loc='upper right')
    # plt.show()

    # 训练
    pp = Perceptron(eta=.1)
    pp.fit(X, y)
    plot_decision_regions(X, y, pp)
    pass


if __name__ == '__main__':
    main()
