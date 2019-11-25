#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hao Wang
@since: 2019/5/21
@function: define the setting arguments, paths, directories or other settings
"""
import tensorflow as tf


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.flags.DEFINE_integer('batch_size', 50, 'number of example per batch')
tf.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
tf.flags.DEFINE_integer('max_sentence_len', 40, 'max number of words per sentence')
tf.flags.DEFINE_integer('max_doc_len', 1, 'max number of sentences per doc')
tf.flags.DEFINE_float('max_grad_norm', 5.0, 'maximal gradient norm')
tf.flags.DEFINE_float('l1_reg', 0.001, 'l1 regularization')
tf.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.flags.DEFINE_integer('display_step', 50, 'number of test display step')
tf.flags.DEFINE_integer('n_epoch', 100, 'number of epoch')
tf.flags.DEFINE_float('keep_prob1', 0.5, 'dropout keep prob in data_layer')
tf.flags.DEFINE_float('keep_prob2', 1.0, 'dropout keep prob in softmax_layer')
tf.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.flags.DEFINE_bool('pre_trained', 'True', 'whether has pre-trained embedding')
tf.flags.DEFINE_string('embedding_type', 'dynamic', 'embedding type: static or dynamic')
tf.flags.DEFINE_integer('early_stopping', 5, 'the number of early stopping epoch')
tf.flags.DEFINE_integer('decay_steps', 2000, 'decay steps of learning rate')
tf.flags.DEFINE_float('decay_rate', 0.96, 'decay rate of learning rate')
tf.flags.DEFINE_string('model', 'NetAb', 'models: NetAb')
tf.flags.DEFINE_bool('is_train', 'True', 'training or test')

tf.flags.DEFINE_string('ckpt_path', './ckpts_noisy/', 'the path of saving checkpoints')
tf.flags.DEFINE_string('result_path', './results_noisy/', 'the path of saving results')

tf.flags.DEFINE_string('word2id_path', './data/word2id/', 'the path of word2id')
tf.flags.DEFINE_string('data_path', './data/', 'the path of dataset')

tf.flags.DEFINE_string('gpu', '0', 'choose to use which gpu')
tf.flags.DEFINE_string('dataset', 'movie', 'movie, laptop, restaurant')

# : python -m main -gpu 0 -dataset movie
