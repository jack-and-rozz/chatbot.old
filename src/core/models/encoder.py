# coding: utf-8 
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.util import nest
from core.models import setup_cell
from utils.tf_utils import linear

def merge_state(state):
  if isinstance(state[0], LSTMStateTuple):
    new_c = tf.concat([s.c for s in state], axis=1)
    new_h = tf.concat([s.h for s in state], axis=1)
    state = LSTMStateTuple(c=new_c, h=new_h)
  else:
    state = tf.concat(state, 1)
  return state

class WordEncoder(object):
  def __init__(self, config, w_embeddings, keep_prob,
               activation=tf.nn.tanh, shared_scope=None):
    self.w_embeddings = w_embeddings
    self.keep_prob = keep_prob
    self.shared_scope = shared_scope

  def __call__(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def encode(self, wc_inputs):
    outputs = []
    with tf.variable_scope(self.shared_scope or "WordEncoder"):
      for inputs in wc_inputs:
        if len(inputs.get_shape()) == 3: # char-based
          raise NotImplementedError()
        elif len(inputs.get_shape()) == 2: # word-based
          word_repls = tf.nn.embedding_lookup(self.w_embeddings, inputs)
          outputs.append(word_repls)
        outputs = tf.concat(outputs, axis=-1)
    return tf.nn.dropout(outputs, self.keep_prob) 

class SentenceEncoder(object):
  def __init__(self, config, keep_prob,
               activation=tf.nn.tanh, shared_scope=None):
    self.rnn_size = config.rnn_size
    self.keep_prob = keep_prob
    self.activation = activation
    self.shared_scope = shared_scope
    is_bidirectional = getattr(tf.nn, config.rnn_type) == tf.nn.bidirectional_dynamic_rnn

    with tf.variable_scope('fw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_fw = setup_cell(config.cell_type, config.rnn_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob)
    with tf.variable_scope('bw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_bw = setup_cell(
        config.cell_type, config.rnn_size, 
        num_layers=config.num_layers, keep_prob=self.keep_prob
      ) if is_bidirectional else None

  def __call__(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def encode(self, input_embeddings, sequence_length):
    with tf.variable_scope(self.shared_scope or "SentenceEncoder") as scope:
      if self.cell_bw is not None:
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
        self.cell_fw, self.cell_bw, input_embeddings,
        sequence_length=sequence_length, dtype=tf.float32, scope=scope)
        with tf.variable_scope("outputs"):
          outputs = tf.concat(outputs, 2)
          outputs = linear(outputs, self.rnn_size)
          outputs = tf.nn.dropout(outputs, self.keep_prob)

        with tf.variable_scope("state"):
          state = merge_state(state)
          state = linear(state, self.rnn_size)
      else:
        outputs, state = tf.nn.dynamic_rnn(
          self.cell_fw, input_embeddings,
          sequence_length=sequence_length, dtype=tf.float32, scope=scope)
    return outputs, state
