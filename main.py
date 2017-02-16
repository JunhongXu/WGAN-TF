from src.wgan import DCGAN
import tensorflow as tf
from src.data_loader import DataSet
from src.utilities import read_data, read_mnist_data
import numpy as np
from matplotlib import pyplot as plt
H, W, C = 32, 32, 1
data = read_mnist_data()
with tf.Session() as sess:
    dcgan = DCGAN(sess, 'mnist', g_size=32, d_size=32, image_size=(H, W, C))
    # train
    dcgan.train(data, 150, 100)

    # Evaluate
    z = np.random.normal(0, 1, size=(64, 100))
    images = dcgan.evaluate(z, num_sample=64, is_train=False)
    for i, im in enumerate(images):
        plt.subplot(8, 8, i+1)
        plt.imshow(im.reshape(H, W), cmap='gray')
        plt.gca().axis('off')
    plt.show()
