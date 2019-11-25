#!/usr/env/bin python
# -*- coding: utf-8 -*-
"""
@author: Hao Wang
@since: 2019/5/21
"""
import argparse
import os

import pymongo


def main(arg):
    word2vec_file = os.path.join(arg.input_path, 'glove.840B.300d.txt')
    with pymongo.MongoClient("mongodb://localhost:27017/") as mongo_client:
        # or set to your custom-built mongo client
        db = mongo_client['word2vec']
        coll = db['glove_840B_300d']
        buffer = list()
        count = 0
        with open(word2vec_file, 'r', encoding='utf-8') as fout:
            for line in fout:
                line_s = line.split(' ')
                if len(line_s) != 300 + 1:
                    print(u'a bad word embedding: {}'.format(line_s[0]))
                    continue
                word = line_s[0]
                vec = [float(v) for v in line_s[1:]]
                count += 1
                key = ['word', 'vec']
                value = [word, vec]
                buffer.append(dict(zip(key, value)))
                if len(buffer) >= 10000:
                    coll.insert_many(buffer)
                    print(count)
                    buffer.clear()
            if buffer:
                coll.insert_many(buffer)
                print(len(buffer))
                buffer.clear()
        print('vec size {}'.format(count))


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--input', '-i', type=str, dest='input_path', action='store',
                         default='./data/',
                         help="type the file path of your 'glove.840B.300d.txt'")
    #
    _args = _parser.parse_args()
    exit(main(_args))
