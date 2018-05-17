#!/usr/bin/python
# -*- coding: utf-8 -*-

from learning_demo_04_tool import *
import timeit

# 读取数据
df = pd.read_csv('learning_demo_04_train.csv')
lable = df['TARGET']
df = df.drop(['ID', 'TARGET'], axis=1)

# analysis
start = timeit.default_timer()
df_summary = eda_analysis(
    missSet=[np.nan, 9999999999, -999999], df=df.iloc[:, 0:3])
print('Running Time: {} seconds.').format(timeit.default_timer() - start)
print(df_summary, '\n')
