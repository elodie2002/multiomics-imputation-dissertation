# coding=utf-8
"""GAIN function for imputation."""

import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index


def gain(data_x, gain_parameters):
    """Impute missing values in data_x using GAIN."""
    seed = gain_parameters.get('seed', None)
    if seed is not None:
        np.random.seed(seed)
        tf.set_random_seed(seed)

    # Reset graph so repeated seed runs do not accumulate TF variables.
    tf.reset_default_graph()

    data_m = 1 - np.isnan(data_x)

    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    no, dim = data_x.shape
    h_dim = int(dim)

    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    X = tf.placeholder(tf.float32, shape=[None, dim])
    M = tf.placeholder(tf.float32, shape=[None, dim])
    H = tf.placeholder(tf.float32, shape=[None, dim])

    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    def generator(x, m):
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    def discriminator(x, h):
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    G_sample = generator(X, M)
    Hat_X = X * M + G_sample * (1 - M)
    D_prob = discriminator(Hat_X, H)

    D_loss_temp = -tf.reduce_mean(
        M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8)
    )
    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    try:
        for _ in tqdm(range(iterations), disable=True):
            batch_idx = sample_batch_index(no, batch_size)
            X_mb = norm_data_x[batch_idx, :]
            M_mb = data_m[batch_idx, :]
            Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
            H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
            H_mb = M_mb * H_mb_temp
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            sess.run([D_solver, D_loss_temp], feed_dict={M: M_mb, X: X_mb, H: H_mb})
            sess.run([G_solver, G_loss_temp, MSE_loss], feed_dict={X: X_mb, M: M_mb, H: H_mb})

        Z_mb = uniform_sampler(0, 0.01, no, dim)
        M_mb = data_m
        X_mb = norm_data_x
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
        imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
        imputed_data = renormalization(imputed_data, norm_parameters)
        imputed_data = rounding(imputed_data, data_x)
    finally:
        sess.close()

    return imputed_data
