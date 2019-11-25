#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hao Wang
@since: 2019/5/21
@function:
"""
import tensorflow as tf


def softmax_layer(inputs, n_hidden, random_base, keep_prob, l2_reg, n_class, scope_name='softmax'):
    with tf.variable_scope(scope_name):
        w = tf.get_variable(
            name='softmax_w',
            shape=[n_hidden, n_class],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
        b = tf.get_variable(
            name='softmax_b',
            shape=[n_class],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_class))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    with tf.name_scope('softmax'):
        outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        scores = tf.nn.xw_plus_b(outputs, w, b, 'scores')
    return scores


def transition_layer(inputs, n_hidden, l2_reg, random_base, scope_name='mlp'):
    """
    :param scope_name:
    :param random_base:
    :param l2_reg:
    :param n_hidden:
    :param inputs: batch * n_hidden
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(scope_name):
        w = tf.get_variable(
            name='att_w',
            shape=[n_hidden, n_hidden],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
        b = tf.get_variable(
            name='att_b',
            shape=[n_hidden],
            # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
            initializer=tf.random_uniform_initializer(-0., 0.),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
        u = tf.get_variable(
            name='att_u',
            shape=[n_hidden, 2],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
        )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.tanh(tf.matmul(inputs, w) + b)
    tmp = tf.reshape(tf.matmul(tmp, u), [batch_size, 2])
    tmp_e = tf.exp(tmp)
    _sum = tf.reduce_sum(tmp_e, axis=1, keepdims=True) + 1e-9
    alpha = tmp / _sum
    # alpha = tmp
    return alpha
