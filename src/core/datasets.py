#coding: utf-8
import tensorflow as tf
import sys, re, random, itertools, os
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from core.vocabularies import _BOS, BOS_ID, _PAD, PAD_ID, _NUM,  WordVocabulary, CharVocabulary
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
  def __init__(self, info, w_vocab, c_vocab, wbase=True, cbase=True):

    self.path = info.path
    self.max_lines = info.max_lines if info.max_lines else None
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.wbase = cbase
    self.cbase = wbase
    self.load = False
    self.original = common.dotDict({
      'w_contexts': None,
      'c_contexts': None,
      'responses' : None,
    })
    self.symbolized = common.dotDict({
      'w_contexts': None,
      'c_contexts': None,
      'responses' : None,
    })
    # [batch_size, context_max_len, utterance_max_len]
    # [batch_size, context_max_len, utterance_max_len, word_max_len]

  @classmethod
  def preprocess_dialogue(self_class, dialogue, context_max_len=0):
    speaker_changes = []
    utterances = []
    for turn in dialogue.split(_EOT):
      new_uttrs = self_class.preprocess_turn(turn)
      utterances += new_uttrs
      speaker_changes += [1] + [0 for _ in range(len(new_uttrs)-1)] 
    # keep up to the last 'context_max_len' contexts.
    if context_max_len:
      utterances = utterances[-context_max_len:]
      speaker_changes = speaker_changes[-context_max_len:]
      # We assume a speaker change has occurred at the beginning of every dialogue.
      speaker_changes[0] = 1
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
    return uttr.strip()

  @common.timewatch()
  def load_data(self, context_max_len=0):
    sys.stderr.write('Loading dataset from %s ...\n' % (self.path))
    data = pd.read_csv(self.path, nrows=self.max_lines)

    sys.stderr.write('Preprocessing ...\n')
    if 'Label' in data:
      data = data[data['Label'] == 1]
    contexts, self.speaker_changes = list(zip(*[self.preprocess_dialogue(x, context_max_len=context_max_len) for x in data['Context']]))
    col_response = 'Utterance' if 'Utterance' in data else 'Ground Truth Utterance'
    responses = [self.preprocess_turn(x)[0] for x in data[col_response]]

    if not self.wbase and not self.cbase:
      raise ValueError('Either \'wbase\' or \'cbase\' must be True.')

    # Separate contexts and responses into words (or chars), and convert them into their IDs.
    if self.wbase:
      self.original.w_contexts = [[self.w_vocab.tokenizer(u) for u in context] 
                                  for context in contexts]
      self.symbolized.w_contexts = [[self.w_vocab.sent2id(u) for u in context] 
                                    for context in self.original.w_contexts]
    else:
      self.original.w_contexts = [None for context in contexts] 
      self.symbolized.w_contexts = [None for context in contexts] 

    if self.cbase:
      self.original.c_contexts = [[self.c_vocab.tokenizer(u) for u in context] 
                                  for context in contexts]

      self.symbolized.c_contexts = [[self.c_vocab.sent2id(u) for u in context] 
                                    for context in self.original.c_contexts]
    else:
      self.original.c_contexts = [None for context in contexts]
      self.symbolized.c_contexts = [None for context in contexts]
    self.original.responses = [self.w_vocab.tokenizer(r) for r in responses]
    self.symbolized.responses = [self.w_vocab.sent2id(r) for r in responses]

  def get_batch(self, batch_size, word_max_len=0,
                utterance_max_len=0, context_max_len=0, shuffle=False):
    if not self.load:
      self.load_data(context_max_len=context_max_len) # lazy loading.
      self.load = True

    if self.wbase:
      w_contexts = self.symbolized.w_contexts
    if self.cbase:
      c_contexts = self.symbolized.c_contexts
    responses = self.symbolized.responses
    speaker_changes = self.speaker_changes

    data = [tuple(x) for x in zip(w_contexts, c_contexts, responses, speaker_changes)]
    if shuffle: # For training.
      random.shuffle(data)
    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      batch = [x[1] for x in b]
      w_contexts, c_contexts, responses, speaker_changes = zip(*batch)
      if self.wbase:
        pass
      if self.cbase:
        pass
      responses = tf.keras.preprocessing.sequence.pad_sequences(
        responses, maxlen=utterance_max_len, 
        padding='post', truncating='post', value=PAD_ID)
      yield common.dotDict({
        'w_contexts': w_contexts,
        'c_contexts': c_contexts,
        'responses': responses,
        'speaker_changes': speaker_changes
      })
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
  def create_vocab_from_data(config):
    train_data_path = config.dataset_info.train.path
    w_vocab_size = config.w_vocab_size
    c_vocab_size = config.c_vocab_size
    lowercase = config.lowercase

    w_vocab_path = train_data_path + '.Wvocab' + str(w_vocab_size)
    if lowercase:
      w_vocab_path += '.lower'
    c_vocab_path = train_data_path + '.Cvocab' + str(c_vocab_size)
    if not (os.path.exists(w_vocab_path) and os.path.exists(c_vocab_path)):
      #data = pd.read_csv(train_data_path, nrows=1000)
      data = pd.read_csv(train_data_path)
      data = data[data['Label'] == 1]
      contexts, _ = list(zip(*[_UbuntuDialogueDataset.preprocess_dialogue(x) for x in data['Context']]))
      contexts = common.flatten(contexts)
      responses = [_UbuntuDialogueDataset.preprocess_turn(x) for x in data['Utterance']]
      responses = common.flatten(responses)
      texts = contexts + responses
      texts = common.flatten([l.split() for l in texts])
      if type(texts[0]) == str:
        texts = [word.decode('utf-8') for word in texts] # List of unicode.

    else:
      texts = ['-']

    w_vocab = WordVocabulary(w_vocab_path, texts, 
                             vocab_size=w_vocab_size, lowercase=lowercase)
    c_vocab = CharVocabulary(c_vocab_path, texts, 
                             vocab_size=c_vocab_size, lowercase=False, 
                             normalize_digits=False)
    return w_vocab, c_vocab
