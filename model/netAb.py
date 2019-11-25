#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hao Wang
@since: 2019/5/21
@function:
"""
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

import config
from data_helper import batch_index, load_word2id, load_y2id_id2y, load_word2vector, recover_data_from_files
from model.nn_layer import transition_layer, softmax_layer


class NetAbModel(object):
    def __init__(self, domain, flags, filter_list=(3, 4, 5), filter_num=100):
        self.config = flags
        self.filter_list = filter_list
        self.filter_num = filter_num
        # placeholder
        self.sen_x_batch = None
        self.sent_len_batch = None
        self.sen_y_batch = None
        self.keep_prob1 = None
        self.keep_prob2 = None
        # embedding
        self.add_placeholder()
        self.word2id = None
        # self.id2word = None
        self.vocab_size = None
        self.embedding = None
        inputs = self.add_embedding(domain)
        # model
        self.sen_logits, self.sen_logits2 = self.netAb(inputs)
        # noisy-loss
        self.loss = self.add_loss(self.sen_logits)
        self.accuracy, self.accuracy_num = self.add_accuracy(self.sen_logits)
        self.train_op = self.add_train_op(self.loss)
        # clean-loss
        self.loss2 = self.add_loss(self.sen_logits2)
        self.accuracy2, self.accuracy_num2 = self.add_accuracy(self.sen_logits2)
        self.train_op2 = self.add_train_op(self.loss2)

    def add_placeholder(self):
        self.sen_x_batch = tf.placeholder(tf.int32, [None, self.config.max_sentence_len])
        self.sent_len_batch = tf.placeholder(tf.int32, [None])
        self.sen_y_batch = tf.placeholder(tf.float32, [None, self.config.n_class])
        self.keep_prob1 = tf.placeholder(tf.float32)
        self.keep_prob2 = tf.placeholder(tf.float32)

    def add_embedding(self, domain):
        if self.config.pre_trained:
            self.word2id, w2v = load_word2vector(self.config.word2id_path, domain)
            # self.word2id, self.id2word, w2v = load_w2v_mongo(domain)
        else:
            self.word2id = load_word2id(self.config.word2id_path, domain)
            self.vocab_size = len(self.word2id)
            w2v = tf.random_uniform([self.vocab_size, self.config.embedding_dim], -1.0, 1.0, trainable=True)
        if self.config.embedding_type == 'static':
            self.embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')
        else:
            self.embedding = tf.Variable(w2v, dtype=tf.float32, name='word_embedding')
        inputs = tf.nn.embedding_lookup(self.embedding, self.sen_x_batch)
        return inputs

    def create_feed_dict(self, sen_x_batch, sent_len_batch, sen_y_batch, kp1=1.0, kp2=1.0):
        holder_list = [self.sen_x_batch, self.sent_len_batch, self.sen_y_batch,
                       self.keep_prob1, self.keep_prob2]
        feed_list = [sen_x_batch, sent_len_batch, sen_y_batch, kp1, kp2]
        return dict(zip(holder_list, feed_list))

    # cnn layer
    def add_cnn_layer(self, inputs, inputs_dim, max_len, scope_name='cnn'):
        inputs = tf.expand_dims(inputs, -1)
        pooling_outputs = []
        for i, filter_size in enumerate(self.filter_list):
            ksize = [filter_size, inputs_dim]
            conv = tf.contrib.layers.conv2d(inputs=inputs,
                                            num_outputs=self.filter_num,
                                            kernel_size=ksize,
                                            stride=1,
                                            padding='VALID',
                                            activation_fn=tf.nn.relu,
                                            scope='conv_' + scope_name + str(i))
            ksize = [max_len - filter_size + 1, 1]
            pooling = tf.contrib.layers.max_pool2d(inputs=conv,
                                                   kernel_size=ksize,
                                                   stride=1,
                                                   padding='VALID',
                                                   scope='pooling_' + scope_name)
            pooling_outputs.append(pooling)
        hiddens = tf.concat(pooling_outputs, 3)
        hiddens = tf.reshape(hiddens, [-1, self.filter_num * len(self.filter_list)])
        return hiddens

    # cnn layer
    def add_noisy_cnn_layer(self, inputs, inputs_dim, max_len, scope_name='cnn'):
        inputs = tf.expand_dims(inputs, -1)
        pooling_outputs = []
        for i, filter_size in enumerate(self.filter_list):
            ksize = [filter_size, inputs_dim]
            conv = tf.contrib.layers.conv2d(inputs=inputs,
                                            num_outputs=self.filter_num,
                                            kernel_size=ksize,
                                            stride=1,
                                            padding='VALID',
                                            activation_fn=tf.nn.relu,
                                            scope='conv_' + scope_name + str(i))
            ksize = [max_len - filter_size + 1, 1]
            pooling = tf.contrib.layers.max_pool2d(inputs=conv,
                                                   kernel_size=ksize,
                                                   stride=1,
                                                   padding='VALID',
                                                   scope='pooling_' + scope_name)
            pooling_outputs.append(pooling)
        hiddens = tf.concat(pooling_outputs, 3)
        hiddens = tf.reshape(hiddens, [-1, self.filter_num * len(self.filter_list)])
        return hiddens

    def netAb(self, inputs):
        print('Running NetAb...')
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)
        inputs = tf.reshape(inputs, [-1, self.config.max_sentence_len, self.config.embedding_dim])
        # word-sentence: cnn
        outputs_sen = self.add_cnn_layer(inputs, self.config.embedding_dim, self.config.max_sentence_len, 'h')
        outputs_sen_dim = self.filter_num * len(self.filter_list)
        outputs_sen = tf.reshape(outputs_sen, [-1, outputs_sen_dim])
        noisy_cnn = self.add_noisy_cnn_layer(inputs, self.config.embedding_dim, self.config.max_sentence_len, 'u')
        noisy_cnn = tf.reshape(noisy_cnn, [-1, outputs_sen_dim])
        # fully-connection
        clean_logits = softmax_layer(outputs_sen, outputs_sen_dim, self.config.random_base, self.keep_prob2,
                                     self.config.l2_reg, self.config.n_class, 'sen_softmax')
        p1 = transition_layer(noisy_cnn, outputs_sen_dim, self.config.l2_reg, self.config.random_base, 'p1')
        p2 = transition_layer(noisy_cnn, outputs_sen_dim, self.config.l2_reg, self.config.random_base, 'p2')
        p1 = tf.expand_dims(p1, 2)
        p2 = tf.expand_dims(p2, 2)
        prob = tf.concat([p1, p2], 2)
        sen_logits = tf.expand_dims(clean_logits, 1)
        noisy_logits = tf.squeeze(tf.matmul(sen_logits, prob))
        return noisy_logits, clean_logits

    def add_loss(self, sen_logits):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=sen_logits, labels=self.sen_y_batch)
        self.sen_vars = [var for var in tf.global_variables()
                         if 'h' in var.name or 'u' in var.name or 'p1' in var.name or 'p2' in var.name]
        # print(self.sen_vars)
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='sen_softmax')
        # print(reg_loss)
        loss = tf.reduce_mean(loss)  # TODO+ self.config.l1_reg * tf.add_n(reg_loss)
        return loss

    def add_accuracy(self, scores):
        correct_predicts = tf.equal(tf.argmax(scores, 1), tf.argmax(self.sen_y_batch, 1))
        accuracy_num = tf.reduce_sum(tf.cast(correct_predicts, tf.int32))  # the number of correct predicting docs
        accuracy = tf.reduce_mean(tf.cast(correct_predicts, tf.float32), name='accuracy')  # accuracy metric result
        return accuracy, accuracy_num

    def add_train_op(self, doc_loss):
        # new_learning_rate = current_learning_rate * decay_rate ^ (global_step / decay_steps)
        global_step = tf.Variable(0, name='global_step', trainable=False)  # record the current step (global step)
        self.lr = tf.train.exponential_decay(self.config.lr, global_step, self.config.decay_steps,
                                             self.config.decay_rate, staircase=True)
        # the optimizer used in this work
        # optimizer = tf.train.AdadeltaOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(doc_loss, self.sen_vars), self.config.max_grad_norm)
        train_op = optimizer.apply_gradients(zip(grads, self.sen_vars), name='train_op', global_step=global_step)
        # train_op = optimizer.minimize(doc_loss, global_step=global_step, var_list=self.doc_vars)
        return train_op

    def run_op(self, sess, op, sen_x, sen_len, sen_y, kp1=1.0, kp2=1.0):
        res_list = []
        len_list = []
        for indices in batch_index(len(sen_x), self.config.batch_size, n_iter=1, is_shuffle=False, is_train=False):
            feed_dict = self.create_feed_dict(sen_x[indices], sen_len[indices], sen_y[indices], kp1, kp2)
            res = sess.run(op, feed_dict=feed_dict)
            res_list.append(res)
            len_list.append(len(indices))
        if type(res_list[0]) is list:  # if op is a list
            res = np.concatenate(res_list, axis=1)
        elif op is self.accuracy_num or op is self.accuracy_num2:
            res = sum(res_list)  # sum all batches
        elif op is self.sen_logits or op is self.sen_logits2:
            res = np.concatenate(np.asarray(res_list), 0)
        else:  # for los, etc.
            res = sum(res_list) * 1.0 / len(len_list)
        return res

    def run_cleaner(self, sess, feed_dict):
        sess.run([self.train_op2], feed_dict=feed_dict)

    def pre_run(self, sess, feed_dict):
        sess.run([self.train_op2], feed_dict=feed_dict)

    def run(self, sess, feed_dict):
        logits = sess.run([self.sen_logits2], feed_dict=feed_dict)
        _, loss, acc_num = sess.run([self.train_op, self.loss, self.accuracy_num], feed_dict=feed_dict)
        return loss, acc_num, np.concatenate(np.asarray(logits), 0)


def test_case(sess, classifier, sen_x, sen_len, sen_y):
    score = classifier.run_op(sess, classifier.sen_logits2, sen_x, sen_len, sen_y)
    loss = classifier.run_op(sess, classifier.loss2, sen_x, sen_len, sen_y)
    acc_num = classifier.run_op(sess, classifier.accuracy_num2, sen_x, sen_len, sen_y)
    y_pred = np.argmax(score, axis=1)
    y_true = np.argmax(sen_y, axis=1)
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                                               labels=[0, 1], average=None)
    return acc_num * 1.0 / len(sen_y), loss, f_class[0]


def run_test(sess, classifier, domain, sen_x, sen_len, sen_y):
    scores = classifier.run_op(sess, classifier.sen_logits2, sen_x, sen_len, sen_y)
    acc_num = classifier.run_op(sess, classifier.accuracy_num2, sen_x, sen_len, sen_y)
    y_pred = np.argmax(scores, axis=1)
    y_true = np.argmax(sen_y, axis=1)
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                                               labels=[0, 1], average=None)
    _, id2y = load_y2id_id2y('./data/y2id.txt')
    result_save_path = classifier.config.result_path + classifier.config.model + '/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    with open(result_save_path + domain + '_test.txt', 'w', encoding='utf-8') as fin:
        fin.write('ACC: ' + str(acc_num * 1.0 / len(sen_x)) + '\t')
        fin.write('P: ' + str(p_class) + '\tR: ' + str(r_class) +
                  '\tF1: ' + str(f_class) + '\tF1_macro: ' + str(f_class.mean()) + '\n')
        for id_y in y_pred:
            fin.write(id2y[id_y] + '\n')
    with open(result_save_path + domain + '_true.txt', 'w', encoding='utf-8') as fin:
        for id_y in y_true:
            fin.write(id2y[id_y] + '\n')
    print('Test. Acc = {}, P = {}, R = {}, F1 = {}, F1_macro = {}'.
          format(acc_num * 1.0 / len(sen_x), p_class, r_class, f_class, f_class.mean()))


def train_run(_):
    flags_ = config.FLAGS
    domain = flags_.dataset  # movie, laptop, restaurant
    print('{} Learning start: >>>\n'.format(domain))
    tf.reset_default_graph()
    os.environ['CUDA_VISIBLE_DEVICES'] = flags_.gpu
    classifier = NetAbModel(domain, flags_)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # gpu_config.gpu_options.allow_growth = True
    gpu_config.allow_soft_placement = False  # If 'True': allow cpu, if no gpu
    saver = tf.train.Saver(tf.global_variables())
    save_path = classifier.config.ckpt_path + classifier.config.model + '/' + domain + '_ckpt'
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        best_val_acc = 0
        best_val_epoch = 0
        # best_test_acc = 0
        training_path = os.path.join(flags_.data_path, 'TrainingSens/')
        train_sen_x, train_sen_len, train_sen_y = recover_data_from_files(
            training_path, 'training', domain, flags_.max_sentence_len)
        val_path = os.path.join(flags_.data_path, 'ValSens/')
        val_sen_x, val_sen_len, val_sen_y = recover_data_from_files(
            val_path, 'validation', domain, flags_.max_sentence_len)
        test_path = os.path.join(flags_.data_path, 'TestSens/')
        test_sen_x, test_sen_len, test_sen_y = recover_data_from_files(
            test_path, 'test', domain, flags_.max_sentence_len)
        # train_sen_x, train_sen_len, train_sen_y = load_inputs_document_mongo(
        #     domain, 'train_noisy', classifier.word2id, flags_.max_sentence_len, flags_.max_doc_len)
        # val_sen_x, val_sen_len, val_sen_y = load_inputs_document_mongo(
        #     domain, 'dev', classifier.word2id, flags_.max_sentence_len, flags_.max_doc_len)
        # test_sen_x, test_sen_len, test_sen_y = load_inputs_document_mongo(
        #     domain, 'test', classifier.word2id, flags_.max_sentence_len, flags_.max_doc_len)
        if classifier.config.is_train:
            for epoch_i in range(flags_.n_epoch):
                print('=' * 20 + 'Epoch ', epoch_i, '=' * 20)
                total_loss = []
                total_acc_num = []
                total_num = []
                if epoch_i < 5:
                    for step, indices in enumerate(batch_index(len(train_sen_y), flags_.batch_size, n_iter=1), 1):
                        indices = list(indices)

                        feed_dict = classifier.create_feed_dict(train_sen_x[indices], train_sen_len[indices],
                                                                train_sen_y[indices],
                                                                flags_.keep_prob1, flags_.keep_prob2)
                        classifier.pre_run(sess, feed_dict=feed_dict)
                    continue
                for step, indices in enumerate(batch_index(len(train_sen_y), flags_.batch_size, n_iter=1), 1):
                    indices = list(indices)
                    # if epoch_i < 10:
                    feed_dict = classifier.create_feed_dict(train_sen_x[indices], train_sen_len[indices],
                                                            train_sen_y[indices],
                                                            flags_.keep_prob1, flags_.keep_prob2)
                    loss, acc_num, logits = classifier.run(sess, feed_dict=feed_dict)
                    y_pred_set = np.argmax(logits, axis=1)
                    y_true_set = np.argmax(train_sen_y[indices], axis=1)
                    f_indices = np.arange(0, len(indices))
                    valid_indices = f_indices[y_pred_set == y_true_set]
                    indices_new = list(np.array(indices)[valid_indices])
                    if indices_new is None:
                        continue
                    # else:
                    #     indices_new = indices
                    # indices_new = indices
                    feed_dict = classifier.create_feed_dict(train_sen_x[indices_new], train_sen_len[indices_new],
                                                            train_sen_y[indices_new],
                                                            flags_.keep_prob1, flags_.keep_prob2)
                    classifier.run_cleaner(sess, feed_dict=feed_dict)
                    total_loss.append(loss)
                    total_acc_num.append(acc_num)
                    total_num.append(len(indices))
                    verbose = flags_.display_step
                    if step % verbose == 0:
                        print('[INFO] Len {}, Epoch {} - Batch {} : loss = {}, acc = {}'.format(
                            len(indices_new), epoch_i, step, np.mean(total_loss[-verbose:]),
                            sum(total_acc_num[-verbose:]) * 1.0 / sum(total_num[-verbose:])))
                loss = np.mean(total_loss)
                acc = sum(total_acc_num) * 1.0 / sum(total_num)
                print('\n[INFO] Epoch {} : mean loss = {}, mean acc = {}'.format(epoch_i, loss, acc))
                if np.isnan(loss):
                    raise ValueError('[Error] loss is not a number!')
                # validation
                val_acc, val_loss, val_f1 = test_case(sess, classifier, val_sen_x, val_sen_len, val_sen_y)
                print('[INFO] val loss: {}, val acc: {}, val f1: {}'.format(val_loss, val_acc, val_f1))
                # test
                test_acc, test_loss, test_f1 = test_case(sess, classifier, test_sen_x, test_sen_len, test_sen_y)
                print('[INFO] test loss: {}, test acc: {}, test f1: {}'.format(test_loss, test_acc, test_f1))
                print('=' * 25 + ' end', '=' * 25 + '\n')
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = epoch_i
                    # best_test_acc = test_acc
                    if not os.path.exists(classifier.config.ckpt_path + classifier.config.model + '/'):
                        os.makedirs(classifier.config.ckpt_path + classifier.config.model + '/')
                    saver.save(sess, save_path=save_path)
                if epoch_i - best_val_epoch > classifier.config.early_stopping:
                    # here early_stopping is 5 :> 'the number of early stopping epoch'
                    print('Normal early stop at {}!'.format(best_val_epoch))
                    # break
            print('Best val acc = {}'.format(best_val_acc))
            # print('Test acc = {}'.format(best_test_acc))
            best_val_epoch_save_path = classifier.config.result_path + classifier.config.model + '/'
            if not os.path.exists(best_val_epoch_save_path):
                os.makedirs(best_val_epoch_save_path)
            with open(best_val_epoch_save_path + domain + '_bestEpoch.txt', 'w', encoding='utf-8') as fin:
                fin.write('Best epoch: ' + str(best_val_epoch) + '\n')

            saver.restore(sess, save_path)
            print('Model restored from %s' % save_path)
            # # test now
            run_test(sess, classifier, domain, test_sen_x, test_sen_len, test_sen_y)
        else:
            saver.restore(sess, save_path)
            print('Model restored from %s' % save_path)
            # # test now
            run_test(sess, classifier, domain, test_sen_x, test_sen_len, test_sen_y)
        print('Domain {} is done..'.format(domain))
    print('\nTraining complete!\n')


if __name__ == '__main__':
    tf.app.run(train_run)
