from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist


def TFRcordWriter():
    save_dir = "/tmp/mnist"

    # Download data to save_dir
    data_sets = mnist.read_data_sets(save_dir,
                                     dtype=tf.uint8,
                                     reshape=False,
                                     validation_size=1000)

    data_splits = ["train", "test", "validation"]
    for d in range(len(data_splits)):
        print("saving" + data_splits[d])
        data_set = data_sets[d]
        filename = os.path.join(save_dir, data_splits[d] + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(data_set.images.shape[0]):
            image = data_set.images[index].tostring()


# Multi-threading

import threading
import time

gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enqueue = queue.enqueue(gen_random_normal)


def add():
    with tf.Session() as sess:
        for i in range(10):
            sess.run(enqueue)


threads = [threading.Thread(target=add, args=()) for i in range(10)]
