#coding: utf-8
import tensorflow as tf
import sys, re, random, itertools, os
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from core.vocabularies import _BOS, BOS_ID, _PAD, PAD_ID, _NUM, FeatureVocab, WordVocabulary
from utils import evaluation, tf_utils, common
import datasets as self_module

EMPTY = '-' # token for empty label.

def remove_special_tokens(tokens):
  special_tokens = [_BOS, _PAD]
  return [x for x in tokens if x not in special_tokens]


def ids2tokens(s_tokens, t_tokens, p_idxs):
  outs = []
  preds = []
  inp = [x for x in s_tokens if x not in [_BOS, _PAD]]
  if t_tokens is not None:
    for tt in t_tokens:
      out = ' '.join([x for x in tt if x not in [_BOS, _PAD]])
      outs.append(out)
  for pp in p_idxs:
    pred =  [s_tokens[k] for k in pp if len(s_tokens) > k]
    pred = ' '.join([x for x in pred if x not in [_BOS, _PAD]])
    if not pred:
      pred = EMPTY
    preds.append(pred)
  return inp, outs, preds

class DatasetBase(object):
  pass

_EOU = '__eou__'
_EOT = '__eot__'
_URL = '__URL__'
_FILEPATH = '__FILEPATH__'

class _UbuntuDialogueDataset(DatasetBase):
  def __init__(self, info, vocab):
    self.path = info.path
    self.max_lines = info.max_lines if info.max_lines else None
    self.vocab = vocab

  @classmethod
  def preprocess_dialogue(self_class, dialogue):
    speaker_changes = []
    utterances = []
    for turn in dialogue.split(_EOT):
      new_uttrs = self_class.preprocess_turn(turn)
      utterances += new_uttrs
      speaker_changes += [1] + [0 for _ in range(len(new_uttrs)-1)] 
    return utterances, speaker_changes
  
  @classmethod
  def preprocess_turn(self_class, turn):
    return [self_class.preprocess_utterance(uttr) for uttr in turn.split(_EOU) if uttr.strip()]

  @classmethod
  def preprocess_utterance(self_class, uttr):
    def _replace_pattern(uttr, before, after):
      m = re.search(before, uttr)
      if m:
        uttr = uttr.replace(m.group(0), after)
      return uttr

    patterns = [
      ('https?\s*:\s*/\S+/\S*', _URL),
      ('[~.]?/\S*', _FILEPATH),
    ]
    for before, after in patterns:
      uttr = _replace_pattern(uttr, before, after)

    separate_tokens = ['*']
    for t in separate_tokens:
      uttr = uttr.replace(t, t + ' ')
    return uttr.strip().split()
 
  def load_data(self, num_max_lines=None):
    sys.stderr.write('Loading dataset from %s ...\n' % (self.path))
    data = pd.read_csv(self.path, nrows=self.max_lines)
    if 'Label' in data:
      data = data[data['Label'] == 1]
    self.contexts, self.speaker_changes = list(zip(*[self.preprocess_dialogue(x) for x in data['Context']]))
    self.responses = [self.preprocess_turn(x)[0] for x in data['Context']]

  def get_batch(self, batch_size,
                utterance_max_len=None, context_max_len=None, shuffle=False):

    self.load_data() # lazy loading.
    return None

    #return self.tensorized
    # sources, targets = self.symbolized
    # if input_max_len:
    #   paired = [(s,t) for s,t in zip(sources, targets) if not len(s) > input_max_len ]
    #   sources, targets = list(zip(*paired))

    # sources =  tf.keras.preprocessing.sequence.pad_sequences(sources, maxlen=input_max_len, padding='post', truncating='post', value=PAD_ID)
    # targets = list(zip(*targets)) # to column-major. (for padding)
    # targets = [tf.keras.preprocessing.sequence.pad_sequences(targets_by_column, maxlen=output_max_len, padding='post', truncating='post', value=PAD_ID) for targets_by_column in targets]
    # targets = list(zip(*targets)) # to idx-major. (for shuffling)

    # data = [tuple(x) for x in zip(sources, targets, self.original_sources)]

    # if shuffle: # For training.
    #   random.shuffle(data)
    # for i, b in itertools.groupby(enumerate(data), 
    #                               lambda x: x[0] // (batch_size)):
    #   batch = [x[1] for x in b]
    #   b_sources, b_targets, b_ori_sources = zip(*batch)
    #   b_targets = list(zip(*b_targets)) # to column-major.
    #   yield common.dotDict({
    #     'sources': np.array(b_sources),
    #     # Include only the labels in 'target_columns' to batch.
    #     'targets': [np.array(t) for t, col in zip(b_targets, self.all_columns) if col in self.target_columns],
    #     'original_sources': b_ori_sources,
    #     #'original_targets': b_ori_targets,
    #   })


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
  @common.timewatch()
  def __init__(self, info, *args, **kwargs):
    dataset_type = getattr(self_module, '_' + self.__class__.__name__)
    self.train = dataset_type(info.train, *args, **kwargs) 
    self.valid = dataset_type(info.valid, *args, **kwargs)
    self.test = dataset_type(info.test, *args, **kwargs)

class UbuntuDialogueDataset(PackedDatasetBase):
  @staticmethod
  def create_vocab_from_data(data_info, vocab_size, lowercase):
    #data = pd.read_csv(data_info.path, nrows=data_info.max_lines)
    data = pd.read_csv(data_info.path)
    data = data[data['Label'] == 1]
    contexts, _ = list(zip(*[_UbuntuDialogueDataset.preprocess_dialogue(x) for x in data['Context']]))
    contexts = common.flatten(contexts, depth=2)
    responses = [_UbuntuDialogueDataset.preprocess_turn(x) for x in data['Utterance']]
    responses = common.flatten(responses, depth=2)
    texts = contexts + responses
    vocab_path = data_info.path + '.Wvocab' + str(vocab_size)
    if lowercase:
      vocab_path += '.lower'
    
    return WordVocabulary(vocab_path, texts, 
                          vocab_size=vocab_size, lowercase=lowercase)

