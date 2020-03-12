import tensorflow as tf
import numpy as np
'''
x=[ 64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]
y=[ 62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]
计算：
'''

x = tf.constant(np.array([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]))
y = tf.constant(np.array([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]))

sum1 = tf.constant(0, dtype=tf.float64)
sum2 = tf.constant(0, dtype=tf.float64)

for i in range(10):
    sum1 += 10 * x[i] * y[i]
    sum2 += 10 * x[i] * x[i]

sum1 -= (tf.reduce_sum(x)) * (tf.reduce_sum(y))
sum2 -= (tf.reduce_sum(x)) * (tf.reduce_sum(x))
w = sum1 / sum2
b = (tf.reduce_sum(y)- (w * tf.reduce_sum(x)))/10
print("w:")
print(w)
print("b:")
print(b)

