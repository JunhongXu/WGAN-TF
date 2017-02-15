import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from src.data_loader import DataSet
import os
from scipy.misc import imresize
from scipy.misc import imread


def lkrelu(x, alpha=0.02, scope="leaky_relu"):
    with tf.name_scope(scope):
        y = tf.maximum(x, alpha * x)
    return y


def read_mnist_data(reshape=False):
    """
    Read mnist data
    """
    mnist = input_data.read_data_sets('data/mnist', reshape=reshape)
    data = mnist.train._images
    if not reshape:
        # pad the mnist data to 32*32
        data = np.pad(data, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0.0)
    return DataSet(data)


def read_data(directory, H, W, reshape=False):
    """
    Read the image data from given directory
    """
    dirs = os.listdir(directory)
    print("The number of image files is %s" % len(dirs))
    images = []
    for index, image in enumerate(dirs):
        img = imread(os.path.join(directory, image))
        # crop the image if h and w are not equal
        h, w, c = img.shape
        diff = h - w
        if diff < 0:
            img = img[:, abs(diff)//2:w-abs(diff), :]
        else:
            img = img[diff//2:h-abs(diff), :, :]
        # resize the image
        img = imresize(img, size=(H, W))
        images.append(img)
        print("Extracting image number %s" % index)
    return DataSet(np.array(images)/127.5 - 1)


def recover(image):
    return (image + 1.) * 2.
