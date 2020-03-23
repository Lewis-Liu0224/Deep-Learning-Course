import numpy as np
import tensorflow as tf
'''
1. 矩阵运算部分采用TensorFlow实现，数据加载、输入、输出等可以根据需要采用Python/NumPy来实现。
2. 提示用户输入商品房面积和房间数，并进行输入校验。合理的输入如下：
面积：20-500之间的实数
房间数：1-10之间的整数
如果输入正确，根据模型估计房价，并显示。
如果输入数据类型错误，或者输入数据范围不合理，根据错误类型提示，并等待用户重新输入。
'''
print("请输入房屋面积(20-500)和房间数(1-10):")
test_sqt = float(input())
while(test_sqt < 20 or test_sqt> 500):
    print("房屋面积格式不正确，请重新输入")
    test_sqt = float(input())
test_room = int(input())
while(test_room < 1 or test_room > 10):
    print("房屋面积格式不正确，请重新输入")
    test_room = int(input())

X0 = tf.ones(16)
X1 = tf.constant(np.array(
    [137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00, 106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26,
     86.21]))
X2 = tf.constant(np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2]))

X = np.stack((X0, X1, X2), axis=1)

Yn = np.array(
    [145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00, 62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69,
     95.30])

Y = tf.constant(Yn.reshape(-1, 1))

Xt = tf.transpose(X)
W = tf.matmul(tf.matmul(tf.linalg.inv(np.matmul(Xt, X)), Xt), Y)

y_pre = W[1] * test_sqt + W[2] * test_room + W[0]

result = np.array(y_pre)

print("该房屋预测的价格为：", np.round(result[0], 2), "万元")




