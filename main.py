from src.wgan import DCGAN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
H, W, C = 32, 32, 1
data = input_data.read_data_sets('MNIST_data', reshape=False)
with tf.Session() as sess:
    dcgan = DCGAN(sess, 'test', g_size=32, d_size=32, image_size=(H, W, C))
    dcgan.train(data.train, 150000, 100)

