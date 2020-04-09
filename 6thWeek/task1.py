import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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
x_data = tf.linspace(0.0, 1.0, 500)
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


# plt.scatter(x_data,y_data)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.plot(x_data,x_data * 3.1234 + 2.98, 'r', linewidth = 3)
# plt.show()
# w = tf.Variable(np.random.randn(), tf.float32)
# b = tf.Variable(0.0, tf.float32)