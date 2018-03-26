#coding: utf-8
import tensorflow as tf
import sys, re, random, itertools, os
import pandas as pd
from collections import OrderedDict, Counter
from core.vocabularies import _BOS, BOS_ID, _PAD, PAD_ID, _NUM,  WordVocabulary, CharVocabulary
from utils import evaluation, tf_utils, common
#import core.datasets.base as self_module

class DatasetBase(object):
  def __init__(self, info, w_vocab, c_vocab):

    self.path = info.path
    self.max_lines = info.max_lines if info.max_lines else None
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.wbase = w_vocab is not None
    self.cbase = c_vocab is not None
    self.load = False

_EOU = '__eou__'
_EOT = '__eot__'
_URL = '__URL__'
_FILEPATH = '__FILEPATH__'

###################################################
#    Classes for dataset pair (train, valid, test)
###################################################

class PackedDatasetBase(object):
  '''
  The class contains train, valid, test dataset.
  Each dataset class has different types of .
  args:
     dataset_type: A string. It is the name of dataset class defined in config.
     pathes: A list of string. ([train_path, valid_path, test_path])
  kwargs:
     num_train_data: The upperbound of the number of training examples. If 0, all of the data will be used.
     no_train: whether to omit to load training data to save time. (in testing)
  '''
  dataset_type = None
  @common.timewatch()
  def __init__(self, info, *args, **kwargs):
    if not self.dataset_type:
      raise ValueError('The derivative of PackedDatasetBase must have class variable \'\dataset_type\'.')
    dataset_type = self.dataset_type
    #dataset_type = getattr(core.datasets, '_' + self.__class__.__name__)
    self.train = dataset_type(info.train, *args, **kwargs) 
    self.valid = dataset_type(info.valid, *args, **kwargs)
    self.test = dataset_type(info.test, *args, **kwargs)

  @classmethod
  def create_vocab_from_data(self, config):
    train_data_path = config.dataset_info.train.path
    w_vocab_size = config.w_vocab_size
    c_vocab_size = config.c_vocab_size
    lowercase = config.lowercase

    w_vocab_path = train_data_path + '.Wvocab' + str(w_vocab_size)
    if lowercase:
      w_vocab_path += '.lower'
    c_vocab_path = train_data_path + '.Cvocab' + str(c_vocab_size)
    if not (os.path.exists(w_vocab_path) and os.path.exists(c_vocab_path)):
      words = self.get_words(train_data_path)
    else:
      words = ['-']

    w_vocab = WordVocabulary(w_vocab_path, words, vocab_size=w_vocab_size, lowercase=lowercase) if config.wbase else None
    c_vocab = CharVocabulary(c_vocab_path, words, vocab_size=c_vocab_size, lowercase=False, normalize_digits=False) if config.cbase else None
    return w_vocab, c_vocab

  @classmethod
  def get_words(self, train_data_path):
    raise NotImplementedError


