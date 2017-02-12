from src.wgan import DCGAN
import tensorflow as tf


with tf.Session() as sess:
    dcgan = DCGAN(sess, 'test')
    for v in dcgan.c_params:
        print(v.name)
        print(v.get_shape())

    for v in dcgan.g_params:
        print(v.name)
        print(v.get_shape())