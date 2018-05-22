import tensorflow as tf

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

ArrayTF()