# from cs231n lecture 8
# CNN framework
import numpy as np

np.random.seed(0)
import tensorflow as tf
import matplotlib.pyplot as plt

def testTensorFlow():
    N, D = 3000, 4000

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = tf.placeholder(tf.float32)

    a = x * y
    b = a + z
    c = tf.reduce_sum(b)

    grad_x, grad_y, grad_z = tf.gradients(c, (x, y, z))

    with tf.Session() as sess:
        values = {
            x: np.random.randn(N, D),
            y: np.random.randn(N, D),
            z: np.random.randn(N, D),
        }

        out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values)

        c_val, grad_x_val, grad_y_val, grad_z_val = out
        print(out)


# Change w1 and w2 from placeholder to variable
def testNeuralNet():
    N, D, H = 64, 1000, 100
    x = tf.placeholder(tf.float32, shape=(N, D))
    y = tf.placeholder(tf.float32, shape=(N, D))
    w1 = tf.Variable(tf.random_normal((D, H)))
    w2 = tf.Variable(tf.random_normal((H, D)))

    h = tf.maximum(tf.matmul(x, w1), 0)
    y_pred = tf.matmul(h, w2)
    #diff = y_pred - y
    #loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))
    #grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
    loss = tf.losses.mean_squared_error(y_pred, y)

    #learning_rate = 1e-5
    #new_w1 = w1.assign(w1 - learning_rate * grad_w1)
    #new_w2 = w2.assign(w2 - learning_rate * grad_w2)
    optimizer = tf.train.GradientDescentOptimizer(1e-5)
    updates = optimizer.minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        values = {
            x: np.random.randn(N, D),
            y: np.random.randn(N, D),
        }
        loss_val_array =[]
        for t in range(50):
            loss_val, _ = sess.run([loss, updates], feed_dict=values)
            loss_val_array.append(loss_val)
        #print(loss_val_array)
        plt.plot(loss_val_array)
        plt.show()

def testNeuralNetWithTF():
    N, D, H = 64, 1000, 100
    x = tf.placeholder(tf.float32, shape=(N, D))
    y = tf.placeholder(tf.float32, shape=(N, D))

    init = tf.contrib.layers.xavier_initializer()
    h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=init)
    y_pred = tf.layers.dense(inputs=h,units=D,kernel_initializer=init)
    loss = tf.losses.mean_squared_error(y_pred, y)
    optimizer = tf.train.GradientDescentOptimizer(1e-5)
    updates = optimizer.minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        values = {
            x: np.random.randn(N, D),
            y: np.random.randn(N, D),
        }
        loss_val_array =[]
        for t in range(50000):
            loss_val, _ = sess.run([loss, updates], feed_dict=values)
            loss_val_array.append(loss_val)
        #print(loss_val_array)
        plt.plot(loss_val_array)
        plt.show()
#testNeuralNet()
testNeuralNetWithTF()
