#!/usr/env/bin python
# -*- coding: utf-8 -*-
"""
@author: Hao Wang
@since: 2019/5/21
"""
import tensorflow as tf
from model import netAb


def main():
    # parses the argument flags and then runs the scripts `main` function while passing the flags
    tf.app.run(netAb.train_run)


if __name__ == "__main__":
    exit(main())
