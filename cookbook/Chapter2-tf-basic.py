import tensorflow as tf
import numpy as np

a = tf.constant(1)
b = tf.constant(2)
c = tf.multiply(a, b)
d = tf.add(a, b)
e = tf.subtract(d, c)
f = tf.add(d, c)
g = tf.divide(f, e)


def FirstTF():
    sess = tf.Session()
    outs = sess.run(g)
    sess.close()
    print(outs)

def ArrayTF():
    with tf.Session() as sess:
        fetches = [a,b,c,d,e,f,g]
        outs = sess.run(fetches)
    print(outs)


#ArrayTF()

def ArraysandShapes():
    c = tf.constant(np.array([
        [[1,2,3],
        [4,5,6]],
        [[1,1,1],
         [2,2,2]]
    ]))
    print(format(c.get_shape()))

ArraysandShapes()