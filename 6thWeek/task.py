import tensorflow as tf
import numpy as np


NUM_EAMPLES = 500
# training_inputs = tf.random.normal([NUM_EAMPLES])
training_inputs = tf.linspace(-50.0, 50.0, NUM_EAMPLES)
training_inputs = tf.random.shuffle(training_inputs)
print(training_inputs)
noise = tf.random.normal([NUM_EAMPLES])
training_outputs = training_inputs * 3.1234 + 2.98 + noise


def model(input, weight, bias):
    return input * weight + bias


def loss(weights, biases):
    error = model(training_inputs, weights, biases) - training_outputs
    return tf.reduce_mean(tf.square(error))


def grad(weights, biases):
    with tf.GradientTape() as tape:
        loss_value = loss(weights, biases)
    return tape.gradient(loss_value, [weights, biases])


traing_steps = 2000
learning_rate = 0.01
W = tf.Variable(0.)
B = tf.Variable(0.)

print("Initial loss: {:.3f}".format(loss(W, B)))
for i in range(traing_steps):
    dw, db = grad(W, B)
    W.assign_sub(dw * learning_rate)
    B.assign_sub(db * learning_rate)
    if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))
print("Final loss: {:.3f}".format(loss(W, B)))
print("w = {}, b = {}".format(W.numpy(), B.numpy()))