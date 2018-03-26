# coding:utf-8
import tensorflow as tf
import sys, re, random, itertools, os
import pandas as pd
from collections import OrderedDict, Counter
from core.vocabularies import _BOS, BOS_ID, _PAD, PAD_ID, _NUM,  WordVocabulary, CharVocabulary
from utils import common
from core.datasets.base import DatasetBase, PackedDatasetBase, _EOU, _EOT, _URL, _FILEPATH

# Todo: dialogueを扱う？1対話から1系列のみ取得？それとも刻む？
class _UbuntuDialogueDataset(DatasetBase):
  @classmethod
  def preprocess_dialogue(self, dialogue, context_max_len=0):
    speaker_changes = []
    utterances = []
    for turn in dialogue.split(_EOT):
      new_uttrs = self.preprocess_turn(turn)
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
  def preprocess_turn(self, turn):
    return [self.preprocess_utterance(uttr) for uttr in turn.split(_EOU) if uttr.strip()]

  @classmethod
  def preprocess_utterance(self, uttr):
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
    df = pd.read_csv(self.path, nrows=self.max_lines)

    sys.stderr.write('Preprocessing ...\n')
    if 'Label' in df:
      df = df[df['Label'] == 1]
    contexts, self.speaker_changes = list(zip(*[self.preprocess_dialogue(x, context_max_len=context_max_len) for x in df['Context']]))
    col_response = 'Utterance' if 'Utterance' in df else 'Ground Truth Utterance'
    responses = [self.preprocess_turn(x)[0] for x in df[col_response]]

    if not self.wbase and not self.cbase:
      raise ValueError('Either \'wbase\' or \'cbase\' must be True.')

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
      # TODO: padding
      if self.wbase:
        pass
      if self.cbase:
        pass
      #responses = tf.keras.preprocessing.sequence.pad_sequences(
      #  responses, maxlen=utterance_max_len, 
      #  padding='post', truncating='post', value=PAD_ID)
      yield common.dotDict({
        'w_contexts': w_contexts,
        'c_contexts': c_contexts,
        'responses': responses,
        'speaker_changes': speaker_changes
      })

class UbuntuDialogueDataset(PackedDatasetBase):
  dataset_type = _UbuntuDialogueDataset
  @classmethod
  def get_words(self, train_data_path):
    dataset_type = self.dataset_type
    data = pd.read_csv(train_data_path)
    data = data[data['Label'] == 1]
    contexts, _ = list(zip(*[dataset_type.preprocess_dialogue(x) for x in data['Context']]))
    contexts = common.flatten(contexts)

    responses = [dataset_type.preprocess_turn(x) for x in data['Utterance']]
    responses = common.flatten(responses)
    texts = contexts + responses
    words = common.flatten([l.split() for l in texts])
    if type(texts[0]) == str:
      words = [word.decode('utf-8') for word in words] # List of unicode.
    return words

