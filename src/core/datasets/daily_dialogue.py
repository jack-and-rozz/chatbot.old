# coding:utf-8
import tensorflow as tf
import numpy as np
import sys, re, random, itertools, os
import pandas as pd
from collections import OrderedDict, Counter
from core.vocabularies import _BOS, BOS_ID, _PAD, PAD_ID, _UNK, UNK_ID,  WordVocabulary, CharVocabulary
from utils import common
from core.datasets.base import DatasetBase, PackedDatasetBase, _EOU, _EOT, _URL, _FILEPATH

max_cnn_width=5

def w_dialogue_padding(w_contexts, context_max_len, utterance_max_len):
  # Get maximum length of contexts and utterances.
  _context_max_len = max([len(d) for d in w_contexts])
  context_max_len = context_max_len if context_max_len else _context_max_len
  
  _utterance_max_len = max([max([len(u) for u in d]) for d in w_contexts]) 
  if not utterance_max_len or _utterance_max_len < utterance_max_len:
    utterance_max_len = _utterance_max_len

  # TODO: the sequences encoded by CNN must be longer than the filter size.
  utterance_max_len = max(max_cnn_width, utterance_max_len)

  # Fill empty utterances.
  w_contexts = [[d[i] if i < len(d) else [] for i in xrange(context_max_len)] for d in w_contexts]
  w_contexts = [tf.keras.preprocessing.sequence.pad_sequences(
    d, maxlen=utterance_max_len, 
    padding='post', truncating='post', value=PAD_ID) for d in w_contexts]
  return w_contexts
  
def c_dialogue_padding(c_contexts, context_max_len, utterance_max_len, 
                       word_max_len):
  # Get maximum length of contexts, utterances, and words.
  _context_max_len = max([len(d) for d in c_contexts])
  context_max_len = context_max_len if context_max_len else _context_max_len

  _utterance_max_len = max([max([len(u) for u in d]) for d in c_contexts]) 
  if not utterance_max_len or _utterance_max_len < utterance_max_len:
    utterance_max_len = _utterance_max_len
  _word_max_len = max([max([max([len(w) for w in u]) for u in d]) for d in c_contexts])
  # TODO: the sequences encoded by CNN must be longer than the filter size.
  utterance_max_len = max(max_cnn_width, utterance_max_len)
  word_max_len = max(max_cnn_width, word_max_len)

  if not word_max_len or _word_max_len < word_max_len:
    word_max_len = _word_max_len

  # Fill empty utterances.
  c_contexts = [[d[i] if i < len(d) else [] for i in xrange(context_max_len)] for d in c_contexts]
  c_contexts = [[[u[i] if i < len(u) else [] for i in xrange(utterance_max_len)] for u in d] for d in c_contexts]

  c_contexts = [[tf.keras.preprocessing.sequence.pad_sequences(
    u, maxlen=word_max_len, padding='post', truncating='post',
    value=PAD_ID) for u in d] for d in c_contexts]
  return c_contexts

class _DailyDialogDataset(DatasetBase):
  def __init__(self, info, w_vocab, c_vocab, context_max_len=0):
    self.context_max_len = context_max_len
    DatasetBase.__init__(self, info, w_vocab, c_vocab)

  def preprocess(self, df):
    data = []
    for x in df.values:
      d = self.preprocess_dialogue(x, context_max_len=self.context_max_len)
      if d:
        data.append(d)
    data = common.flatten(data)
    dialogues, acts, emotions, speaker_changes, topics = list(zip(*data))
    contexts, responses, speaker_changes = zip(*[(d[:-1], d[-1], sc[:-1]) for d, sc in zip(dialogues, speaker_changes) if sc[-1] == 1])
    return contexts, responses, speaker_changes

  @classmethod
  def preprocess_dialogue(self, line, context_max_len=0, split_turn=True):
    idx, dialogue, act, emotion, topic = line
    dialogue = [self.preprocess_turn(x.strip(), split_turn) 
                for x in dialogue.split(_EOU) if x.strip()]
    act = [[int(a) for _ in xrange(len(d))] for a, d in zip(act.split(), dialogue)]
    emotion = [[int(e) for _ in xrange(len(d))] for e, d in zip(emotion.split(), dialogue)]
    speaker_change = [[1 if i == 0 else 0 for i in xrange(len(d))] for d in dialogue]

    dialogue = common.flatten(dialogue)
    act = common.flatten(act)
    emotion = common.flatten(emotion)
    speaker_change = common.flatten(speaker_change)

    # The length of the dialogue and its labels must be same.
    if len(set([len(dialogue), len(act), len(emotion)])) == 1:
      # The maximum length of a dialogue is context_max_len + 1 (response).
      dialogue_max_len = context_max_len + 1 if context_max_len else 0
      if not dialogue_max_len or len(dialogue) < dialogue_max_len:
        return [(dialogue, act, emotion, speaker_change, topic)]
      else: # Slice the dialogue.
        dlen = dialogue_max_len
        return [(dialogue[i:i+dlen], act[i:i+dlen], emotion[i:i+dlen], speaker_change[i:i+dlen], topic) for i in xrange(len(dialogue)+1-dialogue_max_len)]
    else:
      return None
  @classmethod
  def preprocess_turn(self, turn, split_turn):
    turn = turn.replace('...', '.')
    if split_turn:
      split_tokens = ['.', '?', '!']
      for st in split_tokens:
        #turn = common.flatten([u.split(' ' + st) for u in turn])
        turn = turn.split(' ' + st)
        turn = [u + ' ' + st for u in turn[:len(turn)-1]] + [turn[-1]]
        turn = (" " + _EOU + " ").join(turn)
      turn = turn.split(_EOU)
    else:
      turn = [turn]
    turn = [self.preprocess_utterance(u.strip()) for u in turn if u.strip()]
    return turn

  @classmethod
  def preprocess_utterance(self, uttr):
    # Make the way of tokenization equivalent to that of the pretrained embeddings.
    replacements = [
      ("’", "'"),
      ("'", " ' "),
      (".", " . "),
      ("-", " - ")
    ]
    for x, y in replacements:
      uttr = uttr.replace(x, y)
    
    return uttr

  @property
  def size(self):
    if not self.load:
      return None
    return len(self.original.responses)

  @property
  def oov_rate(self):
    if not self.load:
      return None
    context_tokens = common.flatten(self.symbolized.w_contexts, depth=2)
    response_tokens = common.flatten(self.symbolized.responses)
    context_tokens = Counter(context_tokens)
    response_tokens = Counter(response_tokens)
    context_unk_rate = 1.0 * context_tokens[UNK_ID] / sum(context_tokens.values())
    response_unk_rate = 1.0 * response_tokens[UNK_ID] / sum(response_tokens.values())
    return context_unk_rate, response_unk_rate

  def load_data(self):
    sys.stderr.write('Loading dataset from %s ...\n' % (self.path))
    df = pd.read_csv(self.path, nrows=self.max_lines)

    sys.stderr.write('Preprocessing ...\n')
    contexts, responses, speaker_changes = self.preprocess(df)

    if not self.wbase and not self.cbase:
      raise ValueError('Either \'wbase\' or \'cbase\' must be True.')

    self.speaker_changes = speaker_changes

    # Separate contexts and responses into words (or chars), and convert them into their IDs.
    self.original = common.dotDict({})
    self.symbolized = common.dotDict({})

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
    self.load = True

  def get_batch(self, batch_size, word_max_len=0,
                utterance_max_len=0, shuffle=False):
    if not self.load:
      self.load_data() # lazy loading.

    
    responses = self.symbolized.responses
    w_contexts = self.symbolized.w_contexts 
    c_contexts = self.symbolized.c_contexts if self.cbase else [None for _ in xrange(len(responses))]
    speaker_changes = self.speaker_changes
    data = [tuple(x) for x in zip(w_contexts, c_contexts, responses, speaker_changes, self.original.w_contexts, self.original.responses)]
    if shuffle: # For training.
      random.shuffle(data)
    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      batch = [x[1] for x in b]
      w_contexts, c_contexts, responses, speaker_changes, ori_w_contexts, ori_responses = zip(*batch)

      # Set the maximum length in the batch as *_max_len if it is not given.
      if self.wbase:
        w_contexts = w_dialogue_padding(w_contexts, self.context_max_len, 
                                        utterance_max_len)
      if self.cbase:
        c_contexts = c_dialogue_padding(c_contexts, self.context_max_len,
                                        utterance_max_len, word_max_len)

      _utterance_max_len = max([len(u) for u in responses]) 
      if not utterance_max_len or _utterance_max_len < utterance_max_len:
        utterance_max_len = _utterance_max_len
      # TODO: the sequences encoded by CNN must be longer than the filter size.
      utterance_max_len = max(max_cnn_width, utterance_max_len)

      responses = tf.keras.preprocessing.sequence.pad_sequences(
        responses, maxlen=utterance_max_len, 
        padding='post', truncating='post', value=PAD_ID)
      speaker_changes = tf.keras.preprocessing.sequence.pad_sequences(
        speaker_changes, maxlen=self.context_max_len,
        padding='post', truncating='post', value=PAD_ID)
      yield common.dotDict({
        'ori_w_contexts': ori_w_contexts,
        'ori_responses': ori_responses,
        'w_contexts': w_contexts,
        'c_contexts': c_contexts,
        'responses': responses,
        'speaker_changes': speaker_changes,
      })

class DailyDialogDataset(PackedDatasetBase):
  dataset_type = _DailyDialogDataset
  @classmethod
  def get_words(self, train_data_path):
    df = pd.read_csv(train_data_path)
    data = self.dataset_type.preprocess(df, context_max_len=0)
    dialogues, _, _, _ = list(zip(*data))
    words = common.flatten([utterance.split() for utterance in common.flatten(dialogues)])
    return words

