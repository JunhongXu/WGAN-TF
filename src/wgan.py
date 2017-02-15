import tensorflow as tf
from tensorflow.contrib import layers as layers
import os
from src.utilities import *
import numpy as np
from scipy.misc import imsave
# TODO: Finish MLP


class GAN(object):
    def __init__(self, sess, name, batch_size=128, image_size=(32, 32, 3), z_dim=100, g_size=64, d_size=64,
                 optimizer=tf.train.RMSPropOptimizer, critic_lr=5e-5, generator_lr=5e-5, clip=0.01, n_critic=5):
        """
        Parameters:
            sess: tf.Session()
                tensorflow session
            name: str
                name of the dataset this model trained on. used for creating log and saving directories
            batch_size: int
                batch size for the model
            image_size: tuple
                the dimension of input images
            z_dim: int
                the dimension of input to generator
            optimizer: tf.train.Optimizer
                optimizer for critic and generator
            critic_lr: float
                learning rate for critic
            generator_lr: float
                learning rate for generator
            clip: float
                clipping the weights in a fixed range
            n_critic: int
                number of iteration to update the critic per iteration
        """
        # model
        self.sess = sess
        self.H, self.W, self.C = image_size
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.optimizer = optimizer
        self.critic_lr = critic_lr
        self.generator_lr = generator_lr
        self.name = name
        self.g_size = g_size
        self.d_size = d_size
        self.clip = clip
        self.n_critic = n_critic

        with tf.variable_scope("step"):
            self.global_step = tf.Variable(name="global_step", dtype=tf.int32, initial_value=0)
            self.global_epoch = tf.Variable(name="global_epoch", dtype=tf.int32, initial_value=0)

        self.images_train_dir, self.images_test_dir, self.log_dir, self.model_dir = self._create_dir()

        # input to critic and generator
        self.x = tf.placeholder(dtype=tf.float32, name='real_data', shape=(self.batch_size, self.H, self.W, self.C))
        self.z = tf.placeholder(dtype=tf.float32, name='g_input', shape=(self.batch_size, self.z_dim))

        # critic output score on real data
        self.real_data_score = self.critic(self.x, reuse=None, is_train=True)

        # generator output
        self.fake_data = self.generator(None, True)

        # critic output score on fake data
        self.fake_data_score = self.critic(self.fake_data, reuse=True, is_train=True)

        # evaluate generator
        self.sample = self.generator(True, False)

        self.g_params = [v for v in tf.trainable_variables() if 'generator' in v.name]
        self.c_params = [v for v in tf.trainable_variables() if "critic" in v.name]

    def train(self, data, max_epoch):
        NotImplementedError()

    def critic(self, x, reuse, is_train):
        NotImplementedError()

    def generator(self, reuse, is_train):
        NotImplementedError()

    def build(self):
        NotImplementedError("Should be implemented in DCGAN or MLP")

    def _save(self):
        pass

    def _load(self):
        pass

    def _create_dir(self):
        """
            Create directories for saving the model.
                1. Saved training images: images/self.name/train
                2. Saved testing images: images/self.name/test
                3. Saved tensorboard log files: log/self.name
                4. Saved tensorflow models: /models/self.name
        """
        images_train_dir = os.path.join('images', self.name, 'train')
        images_test_dir = os.path.join('images', self.name, 'test')
        log_dir = os.path.join('log', self.name)
        model_dir = os.path.join('checkpoint', self.name)
        if not os.path.exists(images_train_dir):
            os.makedirs(images_train_dir)

        if not os.path.exists(images_test_dir):
            os.makedirs(images_test_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        return images_train_dir, images_test_dir, log_dir, model_dir


class DCGAN(GAN):
    """
    Implementation of WGAN. Original paper can be found at "https://arxiv.org/abs/1701.07875"
    """
    def __init__(self, sess, name, batch_size=64, image_size=(32, 32, 3), z_dim=100, g_size=64, d_size=64,
                 optimizer=tf.train.RMSPropOptimizer, critic_lr=5e-5, generator_lr=5e-5):

        super(DCGAN, self).__init__(sess, name, batch_size, image_size, z_dim, g_size, d_size,
                                    optimizer, critic_lr, generator_lr)
        self.weight_clipping = self._weight_clipping_op()

        self.opt_c, self.opt_g, self.critic_loss, self.g_loss = self.build()

        self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter(self.log_dir, graph=sess.graph)
        self.saver = tf.train.Saver()

        # generator summary
        g_grad_summ = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='generator')
        self.g_summ = tf.summary.merge(g_grad_summ + [tf.summary.scalar('generator_loss', self.g_loss)],
                                       name='g_summary')
        # critic summary
        c_grad_summ = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='critic')
        self.c_summ = tf.summary.merge(c_grad_summ + [tf.summary.scalar('critic_loss', self.critic_loss)],
                                       name='c_summary')

    def critic(self, x, reuse, is_train):
        # normalization parameters used for batch normalization
        normalization_params = {'is_training': is_train, 'decay': 0.9, 'updates_collections': None}
        weight_initializer = tf.random_normal_initializer(0, 0.02)
        with tf.variable_scope("critic", reuse=reuse):
            y = layers.conv2d(x, self.g_size, (3, 3), stride=2, activation_fn=lkrelu, normalizer_fn=layers.batch_norm,
                              normalizer_params=normalization_params, scope="conv1",
                              weights_initializer=weight_initializer)

            y = layers.conv2d(y, self.g_size*2, (3, 3), stride=2, activation_fn=lkrelu, normalizer_fn=layers.batch_norm,
                              normalizer_params=normalization_params, scope="conv2",
                              weights_initializer=weight_initializer)

            y = layers.conv2d(y, self.g_size*4, (3, 3), stride=2, activation_fn=lkrelu, normalizer_fn=layers.batch_norm,
                              normalizer_params=normalization_params, scope="conv3",
                              weights_initializer=weight_initializer)

            y = layers.conv2d(y, self.g_size*8, (3, 3), stride=2, activation_fn=lkrelu, normalizer_fn=layers.batch_norm,
                              normalizer_params=normalization_params, scope="conv4",
                              weights_initializer=weight_initializer)

            # the output layer, remove sigmoid function
            y = layers.fully_connected(tf.reshape(y, (self.batch_size, -1)), num_outputs=1, activation_fn=None,
                                       weights_initializer=weight_initializer,
                                       scope="dense_layer")
        return y

    def generator(self, reuse, is_train):
        # normalization parameters used for batch normalization
        normalization_params = {'is_training': is_train, 'decay': 0.9, 'updates_collections': None}
        weight_initializer = tf.random_normal_initializer(0, 0.02)
        with tf.variable_scope('generator', reuse=reuse):

            # first dense layer to convert input z to H/8, W/8, g_size*8
            y = layers.fully_connected(self.z, num_outputs=self.H//8*self.W//8*self.g_size*8, activation_fn=tf.nn.relu,
                                       normalizer_fn=layers.batch_norm, normalizer_params=normalization_params,
                                       scope="dense_layer", weights_initializer=weight_initializer)

            y = tf.reshape(y, (self.batch_size, self.H//8, self.W//8, self.g_size*8))

            y = layers.conv2d_transpose(y, self.g_size*4, 3, stride=2, activation_fn=tf.nn.relu,
                                        normalizer_fn=layers.batch_norm, normalizer_params=normalization_params,
                                        scope="deconv1", weights_initializer=weight_initializer)

            y = layers.conv2d_transpose(y, self.g_size*2, 3, stride=2, activation_fn=tf.nn.relu,
                                        normalizer_fn=layers.batch_norm, normalizer_params=normalization_params,
                                        scope="deconv2", weights_initializer=weight_initializer)

            y = layers.conv2d_transpose(y, self.g_size*1, 3, stride=2, activation_fn=tf.nn.relu,
                                        normalizer_fn=layers.batch_norm, normalizer_params=normalization_params,
                                        scope="deconv3", weights_initializer=weight_initializer)

            y = layers.conv2d_transpose(y, self.C, 3, activation_fn=tf.nn.tanh, stride=1,
                                        weights_initializer=weight_initializer, scope="deconv4")
        return y

    def build(self):
        critic_loss = tf.reduce_mean(self.fake_data_score-self.real_data_score, name="critic_loss")
        opt_c = layers.optimize_loss(critic_loss, self.global_step, optimizer=self.optimizer(self.critic_lr),
                                     variables=self.c_params, learning_rate=self.critic_lr, name='critic_optimizer',
                                     summaries='gradient_norm')

        g_loss = tf.reduce_mean(-self.fake_data_score, name="generator_loss")

        opt_g = layers.optimize_loss(g_loss, self.global_step, optimizer=self.optimizer(self.generator_lr),
                                     variables=self.g_params, learning_rate=self.generator_lr,
                                     name='generator_optimizer', summaries='gradient_norm')
        return opt_c, opt_g, critic_loss, g_loss

    def _weight_clipping_op(self):
        """
            returns the op of critic weight clipping
        """
        with tf.name_scope('weight_clipping'):
            clipped_weights = [tf.assign(var, tf.clip_by_value(var, -self.clip, self.clip))
                               for var in self.c_params]
        return clipped_weights

    def train(self, data, max_epochs, print_every=100):
        gen_iter = 0
        while data.epochs_completed < max_epochs:
            # train critic network for d_iter steps
            if gen_iter < 25 or gen_iter % 500 == 0:
                d_iter = 100
            else:
                d_iter = self.n_critic

            # train critic network
            critic_loss = self._update_critic(data, d_iter)

            # update generator network
            g_loss = self._update_gen(data, gen_iter)

            gen_iter += 1

            if gen_iter % print_every == 0:
                print("[*]At epoch %s/%s, generator loss %s, critic loss %s" % (data.epochs_completed,
                                                                            max_epochs, g_loss, critic_loss))
                # sample images
                z = np.random.normal(0.0, 1.0, size=(self.batch_size, self.z_dim))
                self.evaluate(z, step=gen_iter)

            if gen_iter % 1000 == 0:
                self._save()

    def evaluate(self, z, num_sample=1, is_train=True, step=None):
        assert num_sample < self.batch_size, "%s should be smaller than batch size"
        if is_train:
            save_path = self.images_train_dir
        else:
            save_path = self.images_test_dir

        fake_images = self.sess.run(self.sample, feed_dict={self.z: z})
        # sample from fake images
        random_index = np.random.randint(0, self.batch_size, num_sample)

        fake_images = fake_images[random_index].reshape(num_sample, self.H, self.W) if self.C == 1 else \
                        fake_images[random_index].reshape(num_sample, self.H, self.W, self.C)
        for index, fake_image in enumerate(fake_images):
            imsave(os.path.join(save_path, "%s.png" % step if step is not None else index), fake_image)
        print('[*]Image saving completed.')

    def _save(self):
        self.saver.save(self.sess, os.path.join(self.model_dir, '%s.ckpt' % self.name))
        print('[*]Model saving completed!')

    def _update_critic(self, data, d_iter):
        for i in range(d_iter):
            # get training samples
            batch = data.next_batch(self.batch_size, increment_epoch=False)
            # get input to generator
            z = np.random.normal(0.0, 1.0, size=(self.batch_size, self.z_dim))
            critic_loss, c_summary, _ = self.sess.run([self.critic_loss, self.c_summ, self.opt_c],
                                                      feed_dict={self.x: batch, self.z: z})
            self.writer.add_summary(c_summary)
            # clip the weights
            self.sess.run(self.weight_clipping)
        return critic_loss

    def _update_gen(self, data, gen_iter):
        batch = data.next_batch(self.batch_size)
        z = np.random.normal(0.0, 1.0, size=(self.batch_size, self.z_dim))
        g_loss, g_summary, _ = self.sess.run([self.g_loss, self.g_summ, self.opt_g],
                                                 feed_dict={self.x: batch, self.z: z})
        self.writer.add_summary(g_summary, gen_iter)
        return g_loss
