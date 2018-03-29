# coding: utf-8 
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.util import nest
from core.models.base import setup_cell
from utils.tf_utils import linear, shape, cnn, flatten

def merge_state(state):
  if isinstance(state[0], LSTMStateTuple):
    new_c = tf.concat([s.c for s in state], axis=1)
    new_h = tf.concat([s.h for s in state], axis=1)
    state = LSTMStateTuple(c=new_c, h=new_h)
  else:
    state = tf.concat(state, 1)
  return state

class CNNEncoder(object):
  def __init__(self, config, keep_prob,
               activation=tf.nn.relu, shared_scope=None):
    self.keep_prob = keep_prob
    self.shared_scope = shared_scope
    self.activation = activation

  def __call__(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def encode(self, inputs, sequence_length):
    with tf.variable_scope(self.shared_scope or "CNNEncoder"):
      target_rank = 3 # [*, max_sequence_length, hidden_size]
      flattened_inputs, prev_shape = flatten(inputs, target_rank)
      flattened_aggregated_outputs = cnn(flattened_outputs, 
                                        activation=self.activation)
      target_shape = prev_shape[:-2] + [shape(flattened_aggregated_outputs, -1)]
      outputs = tf.reshape(flattened_aggregated_outputs, target_shape)
    outputs = tf.nn.dropout(outputs, self.keep_prob) 
    return outputs, outputs
CharEncoder = CNNEncoder

class WordEncoder(object):
  def __init__(self, keep_prob, activation=tf.nn.relu, shared_scope=None):
    self.keep_prob = keep_prob
    self.shared_scope = shared_scope

  def __call__(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def encode(self, inputs):
    outputs = []
    with tf.variable_scope(self.shared_scope or "WordEncoder"):
      outputs = inputs
    outputs = tf.nn.dropout(outputs, self.keep_prob) 
    return outputs

class RNNEncoder(object):
  def __init__(self, config, keep_prob,
               activation=tf.nn.relu, shared_scope=None):
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

  def encode(self, inputs, sequence_length):
    with tf.variable_scope(self.shared_scope or "RNNEncoder") as scope:
      # TODO: flatten the tensor with rank >= 4 to rank 3 tensor.
      inputs, prev_shape = flatten(inputs, 3) # [*, max_sequence_length, hidden_size]
      print 'prev_shape',prev_shape
      output_shape = prev_shape[:-2] + [self.rnn_size]
      sequence_length, _ = flatten(sequence_length, 1)

      if self.cell_bw is not None:
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
          self.cell_fw, self.cell_bw, inputs,
          sequence_length=sequence_length, dtype=tf.float32, scope=scope)
        with tf.variable_scope("outputs"):
          outputs = tf.concat(outputs, 2)
          outputs = linear(outputs, self.rnn_size)
          outputs = tf.nn.dropout(outputs, self.keep_prob)

        with tf.variable_scope("state"):
          state = merge_state(state)
          print 'state', state
          state = linear(state, self.rnn_size)
      else:
        outputs, state = tf.nn.dynamic_rnn(
          self.cell_fw, inputs,
          sequence_length=sequence_length, dtype=tf.float32, scope=scope)
      outputs = tf.reshape(outputs, output_shape)
    return outputs, state
