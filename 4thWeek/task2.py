import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
plt.rcParams['font.sans-serif']="SimHei"
plt.figure(figsize=(10, 10))
"""
(1)下载手写数字数据集，读取训练集和测试集数据，放在NumPy数组train_x、train_y、test_x、test_y中；（
train_x：训练集图像，train_y：训练集标签，test_x：测试集图像，test_y：测试集标签）

(2)随机从所有测试集数据中显示16幅数字图像；

(3)16幅图像按照4×4方式排列在一张画布中，每幅图像的子标题
为该图像的标签值，字体大小为14，全局标题为“MNIST测试集样本”，字体大小为20，颜色为红色。
"""

for i in range(1, 17):
    num = np.random.randint(1,50000)
    plt.subplot(5, 4, i+4)
    plt.axis("off")
    plt.imshow(train_x[num], cmap='gray')
    plt.title("标签值："+str(train_y[num]))

plt.suptitle("MNIST测试集样本", y=0.9, fontsize=20, color='red')
plt.show()
