#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf

# 读取数据
ratings_df = pd.read_csv('./data/ratings.csv')
movies_df = pd.read_csv('./data/movies.csv')

# 增加 id（行号）
movies_df['movieRow'] = movies_df.index

# 筛选 movies_df 中的特征
movies_df = movies_df[['movieRow', 'movieId', 'title']]
movies_df.to_csv('./data/movieProcessed.csv', index=False,
                 header=True, encoding='utf-8')

# 将 rating_df 中的 movieId 替换为行号
ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
ratings_df = ratings_df[['userId', 'movieRow', 'rating']]
ratings_df.to_csv('./data/ratingProcessed.csv', index=False,
                  header=True, encoding='utf-8')

# 创建电影评分矩阵 rating 和评分记录矩阵 record
userNo = ratings_df['userId'].max() + 1
movieNo = ratings_df['movieRow'].max() + 1
rating = np.zeros((movieNo, userNo))
flag = 0
rating_df_length = np.shape(ratings_df)[0]
for index, row in ratings_df.iterrows():
    rating[int(row['movieRow']), int(row['userId'])] = row['rating']
    flag += 1
    # print('processed %d, %d left' % (flag, rating_df_length - flag))
record = rating > 0
record = np.array(record, dtype=int)


# 构建模型
def normalizeRatings(rating, record):
    m, n = rating.shape
    rating_mean = np.zeros((m, 1))
    rating_norm = np.zeros((m, n))
    for i in range(m):
        idx = record[i, :] != 0
        rating_mean[i] = np.mean(rating[i, idx])
        rating_norm[i, idx] -= rating_mean[i]
    return np.nan_to_num(rating_norm), np.nan_to_num(rating_mean)


rating_norm, rating_mean = normalizeRatings(rating, record)

num_features = 10
X_parameters = tf.Variable(tf.random_normal(
    [movieNo, num_features], stddev=0.35))
Theta_parmeters = tf.Variable(
    tf.random_normal([userNo, num_features], stddev=.35))
loss = 1/2*tf.reduce_sum(((tf.matmul(X_parameters,
                                     Theta_parmeters, transpose_b=True) - rating_norm) * record) ** 2) + 1/2 * (tf.reduce_sum(X_parameters ** 2) + tf.reduce_sum(Theta_parmeters ** 2))
optimizer = tf.train.AdadeltaOptimizer(1e-4)
train = optimizer.minimize(loss)

# 训练模型
tf.summary.scalar('loss', loss)
summaryMerged = tf.summary.merge_all()
filename = './movie_tensorboard'
writer = tf.summary.FileWriter(filename)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(5000):
    _, movie_summary = sess.run([train, summaryMerged])
    writer.add_summary(movie_summary, i)

# 评估模型
Current_X_parameters, Current_Theta_patameters = sess.run(
    [X_parameters, Theta_parmeters])
predicts = np.dot(Current_X_parameters,
                  Current_Theta_patameters.T) + rating_mean
errors = np.sqrt(np.sum(predicts-rating) ** 2)

# 构建完整的电影推荐系统
user_id = input('请输入用户编号:')
sortedResult = predicts[:, int(user_id)].argsort()[::-1]
idx = 0
print('推荐的评分最高的20部电影是：'.center(80, '='))
for i in sortedResult:
    print('评分：%.2f, 电影名：%s' %
          (predicts[i, int(user_id)], movies_df.iloc[i]['title']))
