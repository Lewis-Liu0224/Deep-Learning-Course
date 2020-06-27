import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale

np.set_printoptions(suppress=True)


def model(x, w, b):
    return tf.multiply(x, w) + b


def loss(x, y, w, b):
    err = model(x, w, b) - y
    squared_err = tf.square(err)
    return tf.reduce_sum(squared_err)


def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])


training_epochs = 10
learning_rate = 0.01
x_data1 = tf.linspace(0.0, 500.0, 500)
x_data = tf.cast(scale(x_data1.numpy()), dtype=tf.float32)
print(x_data)
np.random.seed(5)
y_data = 3.1234 * x_data + 2.98
w = tf.Variable(1.)
b = tf.Variable(0.)
step = 0
loss_list = []
display_step = 20

for epoch in range(training_epochs):
    for xs, ys in zip(x_data, y_data):

        loss_ = loss(xs, ys, w, b)
        loss_list.append(loss_)

        delta_w, delta_b = grad(xs, ys, w, b)
        change_w = delta_w * learning_rate
        change_b = delta_b * learning_rate
        w.assign_sub(change_w)
        b.assign_sub(change_b)

        step = step + 1
        if step % display_step == 0:
            print("Training Epoch:", '%02d' % (epoch + 1), "Step: %03d" % (step),
                  "lose=%.6f" % (loss_),"value of w:%.6f" % (w.numpy()),"value of b:%.6f" % (b.numpy()))
    plt.plot(x_data, w.numpy() * x_data + b.numpy())
plt.show()
print("w:", w.numpy())
print("b:", b.numpy())


#通过训练出的模型预测 x=5.79 时 y 的值，并显示根据目标方程显示的 y 值，
test_x = 5.79
print("当x=5.79时，预测y值为：{:.4f}".format(w*test_x+b))
print("当x=5.79时，目标方程的y值为：{:.4f}".format(3.1234 * test_x + 2.98))

# writer = tf.summary.FileWriter("E:\homework\Deep-Learning-Course\log", tf.get_default_graph())
# writer.close()
# plt.scatter(x_data,y_data)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.plot(x_data,x_data * 3.1234 + 2.98, 'r', linewidth = 3)
# plt.show()
# w = tf.Variable(np.random.randn(), tf.float32)
# b = tf.Variable(0.0, tf.float32)