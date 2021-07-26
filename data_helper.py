#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Hao Wang
@since: 2019/5/21
@function:
"""
import collections
import os
import pickle
from collections import Counter

import numpy as np
import pymongo


def batch_index(length, batch_size, n_iter=1, is_shuffle=True, is_train=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        if is_train:
            batch_num = length // batch_size
        else:
            batch_num = length // batch_size + (1 if length % batch_size else 0)
        for i in range(batch_num):
            yield index[i * batch_size:(i + 1) * batch_size]


class Vocabulary(object):
    """Vocabulary
    """

    EOS = 'UNK'

    def __init__(self, add_eos=True):
        self._add_eos = add_eos
        self._word_dict = None
        self._word_list = None
        self._voc_size = None

    def recover(self, w2id):
        self._word_dict = w2id
        self._voc_size = len(w2id)

    def load(self, iter_voc_item, word_column='word', index_column='index'):
        """Load an existing vocabulary.

        Args:
            iter_voc_item: Iterable object. This can be a list, a generator or a database cursor.
            word_column (str): Column name that contains the word.
            index_column (str): Column name that contains the word index.

        """
        # load word_dict
        word_dict = dict()
        for doc in iter_voc_item:
            word = doc[word_column]
            index = doc[index_column]
            word_dict[word] = index

        # generate word_list
        voc_size = len(word_dict)
        word_list = [None for _ in range(voc_size)]
        for word, index in word_dict.items():
            word_list[index] = word

        self._word_dict = word_dict
        self._word_list = word_list
        self._voc_size = voc_size
        return self

    def dump(self, word_column='word', index_column='index'):
        """Dump the current vocabulary to a dict generator.

        Args:
            word_column (str): Column name for word.
            index_column (str): Column name for index.

        Returns:
            A generator of dict object.

        """
        for word, index in self._word_dict.items():
            yield {
                word_column: word,
                index_column: index
            }

    def generate(self, iter_words, words_column='words', min_count=1, verbose_fn=None):
        """Generate a vocabulary from sentences.

        Args:
            iter_words: Iterable object. This can be a list, a generator or a database cursor.
            words_column (str): Column name that contains "words" data.
            min_count (int): Minimum count of the word in the vocabulary.
            verbose_fn ((int) -> None): Verbose function.
                This is useful when iter_words contains much more documents.

        """
        # statistic info
        counter = collections.defaultdict(int)
        for i, doc in enumerate(iter_words, 1):
            words = doc[words_column]
            for word in words:
                counter[word] += 1
            if verbose_fn:
                verbose_fn(i)
        if '' in counter:
            del counter['']

        # generate word_dict (word -> index)
        word_dict = {self.EOS: 0}
        for word, count in counter.items():
            if count < min_count:
                continue
            index = len(word_dict)
            word_dict[word] = index

        # generate word_list
        voc_size = len(word_dict)
        word_list = [None for _ in range(voc_size)]
        for word, index in word_dict.items():
            word_list[index] = word

        self._word_dict = word_dict
        self._word_list = word_list
        self._voc_size = voc_size
        return self

    @property
    def voc_size(self):
        return self._voc_size

    @property
    def word_dict(self):
        return self._word_dict

    @property
    def word_list(self):
        return self._word_list

    def indexes_to_words(self, indexes):
        id2word = {}
        for index in range(indexes):
            id2word[index] = self._word_list[index]
        return id2word


class WordEmbedding(object):

    def __init__(self):
        self._word_dict = None
        self._word_list = None
        self._emb_mat = None

    def load(self, iter_emb_item, word_column='word', index_column='index', vector_column='vector'):
        # load word_dict and emb_dict
        word_dict = dict()
        emb_dict = dict()
        for doc in iter_emb_item:
            word = doc[word_column]
            index = doc[index_column]
            vector = doc[vector_column]
            word_dict[word] = index
            emb_dict[index] = vector
        voc_size = len(word_dict)

        # generate word_list
        word_list = [None for _ in range(voc_size)]
        for word, index in word_dict.items():
            word_list[index] = word

        # generate emb_list
        emb_list = [None for _ in range(voc_size)]
        for index, vector in emb_dict.items():
            emb_list[index] = vector

        self._word_dict = word_dict
        self._word_list = word_list
        self._emb_mat = np.array(emb_list, np.float32)
        return self

    def dump(self, word_column='word', index_column='index', vector_column='vector'):
        """Dump the current vocabulary to a dict generator.

        Args:
            word_column (str): Column name for word.
            index_column (str): Column name for index.
            vector_column (str): Column name for vector.

        Returns:
            A generator of dict object.

        """
        for word, index in self._word_dict.items():
            vector = self._emb_mat[index]
            yield {
                word_column: word,
                index_column: index,
                vector_column: pickle.dumps(vector)
            }

    def generate(self,
                 voc,
                 iter_pre_trained,
                 word_column='word',
                 vector_column='vector',
                 bound=(-1.0, 1.0),
                 verbose_fn=None):
        """Generate word embedding.

        Args:
            voc (Vocabulary): Vocabulary.
            iter_pre_trained: Iterator/Generator of per-trained word2vec.
            word_column (str): Column name for word.
            vector_column (str): Column name for vector.
            bound (tuple[float]): Bound of the uniform distribution which is used to generate vectors for words that
                not exist in pre-trained word2vec.
            verbose_fn ((int) -> None): Verbose function to indicate progress.

        """
        # inherit input vocabulary's word_dict and word_list
        self._word_dict = voc.word_dict
        self._word_list = voc.word_list

        # generate emb_list
        emb_size = None
        emb_list = [None for _ in range(voc.voc_size)]  # type: list
        for i, doc in enumerate(iter_pre_trained, 1):
            if verbose_fn:
                verbose_fn(i)
            word = doc[word_column]
            vector = doc[vector_column]
            if emb_size is None:
                emb_size = len(vector)
            try:
                index = self._word_dict[word]
            except KeyError:
                continue
            emb_list[index] = vector

        # If a word is not in the pre-trained embeddings, generate a random vector for it
        for i, vector in enumerate(emb_list):
            vector = emb_list[i]
            if vector is None:
                vector = np.random.uniform(bound[0], bound[1], emb_size)
            emb_list[i] = vector

        self._emb_mat = np.array(emb_list, np.float32)
        return self

    @property
    def word_dict(self):
        return self._word_dict

    @property
    def word_list(self):
        return self._word_list

    @property
    def emb_mat(self):
        return self._emb_mat


def load_word2id(word2id_path, domain):
    word2id_file = os.path.join(word2id_path, domain + '_w2id.txt')
    word2id = dict()
    with open(word2id_file, mode='r') as fin:
        for line in fin:
            w2id = line.strip('\n').split('\t')
            word2id[w2id[0]] = int(w2id[1])
    return word2id


def load_word2vector(word2id_path, domain):
    # with pymongo.MongoClient() as conn:
    #     conn['admin'].authenticate('root', 'your password')
    with pymongo.MongoClient("mongodb://localhost:27017/") as conn:
        # TODO: or set to your custom-built mongo client
        # # from mongoDB
        # db = conn['sen']
        # coll_vocab = db['vocab_noisy']
        #
        # print('Loading vocabulary...')
        # voc = Vocabulary()
        # voc.load(
        #     iter_voc_item=(doc for doc in coll_vocab.find({"domain": domain})),
        #     word_column='word',
        #     index_column='value'
        # )
        # word2id = voc.word_dict
        # # id2word = voc.indexes_to_words(voc.voc_size)
        # print(f'Vocabulary loaded. voc_size={voc.voc_size}')
        # from local file
        word2id_file = os.path.join(word2id_path, domain + '_w2id.txt')
        word2id = dict()
        with open(word2id_file, mode='r') as fin:
            for line in fin:
                w2id = line.strip('\n').split('\t')
                word2id[w2id[0]] = int(w2id[1])

        voc = Vocabulary()
        voc.recover(word2id)
        print('Vocabulary loaded. voc_size={}'.format(voc.voc_size))

        print('Generating embeddings...')

        def verbose(i):
            if i % 10000 == 0:
                print(f'Processing {i}', end='\r')

        emb = WordEmbedding()  # glove_840B_300d
        emb.generate(
            voc=voc,
            iter_pre_trained=(
                {'word': doc['word'], 'vec': doc['vec']}
                for doc in conn['word2vec']['glove_840B_300d'].find()
            ),
            word_column='word',
            vector_column='vec',
            verbose_fn=verbose
        )
        w2v = emb.emb_mat

    # return word2id, id2word, w2v
    return word2id, w2v


def change_y_to_onehot(y):
    print(Counter(y))
    class_set = set(y)
    n_class = len(class_set)  # the number of classes
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    print(y_onehot_mapping)
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    with open('./data/y2id.txt', 'w', encoding='utf-8') as fin:
        for k, v in y_onehot_mapping.items():
            fin.write(str(k) + ' ' + str(v) + '\n')
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1  # only tmp[y_onehot_mapping[label]] = 1, others = 0
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_y2id_id2y(file):
    y2id = dict()
    id2y = dict()
    with open(file, 'r', encoding='utf-8') as fout:
        for line in fout:
            y, id_y = line.split()
            y2id[y] = int(id_y)
            id2y[int(id_y)] = y
    return y2id, id2y


def recover_data_from_files(data_path, type_data, domain, max_sen_len):
    data_x_file = os.path.join(data_path, domain + '_x.txt')
    data_y_file = os.path.join(data_path, domain + '_y.txt')
    data_x = []
    data_x_sen = []
    data_y = []
    with open(data_x_file, mode='r') as fin:
        for line in fin:
            index = [int(i) for i in line.strip().split(' ')]
            if len(index) == max_sen_len:
                data_x.append(index)
                data_x_sen.append(max_sen_len)
            else:
                print('data Error!!!')
                raise RuntimeError('dataError')
    with open(data_y_file, mode='r') as fin:
        for line in fin:
            index = int(line.strip().split(' ')[0])
            data_y.append(index)
    data_y = change_y_to_onehot(data_y)
    print('load ' + type_data + ' dataset {} done!'.format(domain))
    return np.asarray(data_x), np.asarray(data_x_sen), np.asarray(data_y)
