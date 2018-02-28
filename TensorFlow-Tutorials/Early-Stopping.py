# matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from tensorflow.examples.tutorials.mnist import input_data
import os
import CNN

saver = tf.train.Saver()

saver_dir = 'checkpoints'

if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)

save_path = os.path.join(saver_dir,'best_validation')

def Init_store():
    CNN.init_variables()
    CNN.print_test_accuracy()
    CNN.optimize(2000)
    saver.save(sess=CNN.session, save_path=save_path)

#Init_store()
saver.restore(sess=CNN.session, save_path=save_path)
CNN.print_test_accuracy()





