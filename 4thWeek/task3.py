import tensorflow as tf
import numpy as np

'''
使用TensorFlow张量运算计算w和b，并输出结果。
已知： 
x=[ 64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]
y=[ 62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]
计算：
其中和分别为x和y的均值，是x中索引值为i的元素，是y中索引值为i的元素。
(3)分别输出W和b的结果。
提示：
正确的输出结果
w= 0.83215.....
b= 10.2340.......
'''
x = tf.constant(np.array([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]))
y = tf.constant(np.array([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]))

xA = tf.reduce_mean(x, axis=0)
yA = tf.reduce_mean(y, axis=0)

sum1 = tf.constant(0, dtype=tf.float64)
sum2 = tf.constant(0, dtype=tf.float64)

for i in range(tf.size(x)):
    a = (x[i]-xA)
    sum1 += a * (y[i]-yA)
    sum2 += a * a

w = sum1/sum2
b = yA-w*xA
print("x:")
print(w)
print("b:")
print(b)
