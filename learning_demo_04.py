#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from scipy import stats
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('./data/learning_demo_04_train.csv')
lable = df['TARGET']
df = df.drop(['ID', 'TARGET'], axis=1)

# Basic Analisis
# 1. missing value
missSet = [np.nan, 9999999999, -999999]

# 2. count distinct
# len(df.iloc[:, 0].unique())
count_un = df.iloc[:, 0:3].apply(lambda x: len(x.unique()))
print(count_un, '\n')

# 3. zero values
# np.sum(df.iloc[:, 0] == 0)
count_zero = df.iloc[:, 0:3].apply(lambda x: np.sum(x == 0))
print(count_zero, '\n')

# 4. mean values
# df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)] # 去除缺失值
# np.mean(df.iloc[:, 0]) # 求均值
df_mean = df.iloc[:, 0:3].apply(lambda x: np.mean(x[~np.isin(x, missSet)]))
print(df_mean, '\n')

# 5. median values
# np.median(df.iloc[:, 0]) # 中位数
df_median = df.iloc[:, 0:3].apply(lambda x: np.median(x[~np.isin(x, missSet)]))
print(df_median, '\n')

# 6. mode values
df_mode = df.iloc[:, 0:3].apply(
    lambda x: stats.mode(x[~np.isin(x, missSet)])[0][0])
df_mode_count = df.iloc[:, 0:3].apply(
    lambda x: stats.mode(x[~np.isin(x, missSet)])[1][0])
df_mode_perct = df_mode_count / df.shape[0]
print(df_mode_perct, '\n')

# 7. min value
df_min = df.iloc[:, 0:3].apply(lambda x: np.min(x[~np.isin(x, missSet)]))
print(df_min, '\n')

# 8. max value
df_max = df.iloc[:, 0:3].apply(lambda x: np.max(x[~np.isin(x, missSet)]))
print(df_max, '\n')

# 9. quantile values
# np.percentile(df.iloc[:, 0][~np.isin(df.iloc[:, 0], missSet)], (1, 5, 25, 50, 75, 95, 99))
json_quantile = {}
for i, name in enumerate(df.iloc[:, 0:3].columns):
    print('the {} columns: {}').format(i, name)
    json_quantile[name] = np.percentile(
        df[name][~np.isin(df[name], missSet)], (1, 5, 25, 50, 75, 95, 99))
df_quantile = pd.DataFrame(json_quantile)[df.iloc[:, 0:3].columns].T
print(df_quantile, '\n')

# 10. frequent values
json_fre_name = {}
json_fre_count = {}


def fill_fre_top_5(x):
    if(len(x)) <= 5:
        new_array = np.full(5, np.nan)
        new_array[0:len(x)] = x
        return new_array


for i, name in enumerate(df.iloc[:, 0:3].columns):
    index_name = df[name][~np.isin(
        df[name], missSet)].value_counts().iloc[0:5, ].index.values
    index_name = fill_fre_top_5(index_name)
    json_fre_name[name] = index_name

    value_count = df[name][~np.isin(
        df[name], missSet)].value_counts().iloc[0:5, ].values
    value_count = fill_fre_top_5(value_count)
    json_fre_count[name] = value_count

df_fre_name = pd.DataFrame(json_fre_name)[df.iloc[:, 0:3].columns].T
df_fre_count = pd.DataFrame(json_fre_count)[df.iloc[:, 0:3].columns].T
df_fre = pd.concat([df_fre_name, df_fre_count], axis=1)
print(df_fre, '\n')

# miss values
df_miss = df.iloc[:, 0:3].apply(lambda x: np.sum(np.isin(x, missSet)))
print(df_miss, '\n')
