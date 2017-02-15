from src.wgan import DCGAN
import tensorflow as tf
from src.data_loader import DataSet
from src.utilities import read_data, read_mnist_data
H, W, C = 32, 32, 1
data = read_mnist_data()
with tf.Session() as sess:
    dcgan = DCGAN(sess, 'mnist', g_size=32, d_size=32, image_size=(H, W, C))
    dcgan.train(data, 25, 100)

