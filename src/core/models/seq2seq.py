# coding:utf-8
import math, sys, time
import numpy as np
from pprint import pprint

import tensorflow as tf
from utils.tf_utils import shape
from core.models import ModelBase, setup_cell
from core.models.encoder import WordEncoder, SentenceEncoder
from core.extensions.pointer import pointer_decoder 
from core.vocabularies import BOS_ID

class Seq2Seq(ModelBase):
  def __init__(self, sess, config, w_vocab, c_vocab):
    ModelBase.__init__(self, sess, config)
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    output_max_len = config.utterance_max_len
    self.is_training = tf.placeholder(tf.bool, [], name='is_training')
    with tf.name_scope('keep_prob'):
      self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate

    # <Sample input>
    # e_inputs: [1, 40, 44, 0, 0], d_outputs: [2, 0, 0] (target=44)
    with tf.name_scope('EncoderInput'):
      self.e_inputs_w_ph = tf.placeholder(
        tf.int32, [None, None], name="EncoderInputWords")
      self.e_inputs_c_ph = tf.placeholder(
        tf.int32, [None, None, None], name="EncoderInputChars")

    with tf.name_scope('batch_size'):
      batch_size = shape(self.e_inputs_w_ph, 0)

    with tf.variable_scope('Embeddings') as scope:
      if w_vocab.embeddings:
        initializer = tf.constant_initializer(w_vocab.embeddings) 
        trainable = config.train_embedding
      else:
        initializer = None
        trainable = True 
      w_embeddings = self.initialize_embeddings(
        'Word', [w_vocab.size, config.w_embedding_size],
        initializer=initializer,
        trainable=trainable)

      if c_vocab.embeddings:
        initializer = tf.constant_initializer(c_vocab.embeddings) 
        trainable = config.train_embedding
      else:
        initializer = None
        trainable = True 
      c_embeddings = self.initialize_embeddings(
        'Char', [c_vocab.size, config.c_embedding_size],
        initializer=initializer,
        trainable=trainable)

    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
      e_inputs_emb = []
      with tf.variable_scope('Word') as scope:
        word_encoder = WordEncoder(w_embeddings, self.keep_prob,
                                   shared_scope=scope)
        e_inputs_emb.append(word_encoder.encode([self.e_inputs_w_ph]))

      with tf.variable_scope('Char') as scope:
        char_encoder = CharEncoder(c_embeddings, self.keep_prob,
                                   shared_scope=scope)
        e_inputs_emb.append(char_encoder.encode([self.e_inputs_c_ph]))

      e_inputs_emb = tf.concat(e_inputs_emb, axis=-1)
      print e_inputs_emb
      with tf.variable_scope('Sent') as scope:
        sent_encoder = SentenceEncoder(config, self.keep_prob, 
                                       shared_scope=scope)
      e_inputs_w_length = tf.count_nonzero(self.e_inputs_w_ph, axis=1)
      e_outputs, e_state = sent_encoder.encode(
        e_inputs_emb, e_inputs_length)
      attention_states = e_outputs

    # self.d_outputs_ph = []
    # self.losses = []
    # self.greedy_predictions = []
    # self.copied_inputs = []
    # for i, col_name in enumerate(config.target_columns):
    #   with tf.name_scope('DecoderOutput%d' % i):
    #     d_outputs_ph = tf.placeholder(
    #       tf.int32, [None, output_max_len], name="DecoderOutput")

    #   ds_name = 'Decoder' if config.share_decoder else 'Decoder%d' % i 
    #   with tf.variable_scope(ds_name) as scope:
    #     d_cell = setup_cell(config.cell_type, config.rnn_size, config.num_layers,
    #                         keep_prob=self.keep_prob)
    #     teacher_forcing = config.teacher_forcing if 'teacher_forcing' in config else False
    #     d_outputs, predictions, copied_inputs = setup_decoder(
    #       d_outputs_ph, e_inputs_emb, e_state, attention_states, d_cell, 
    #       batch_size, output_max_len, scope=scope, 
    #       teacher_forcing=teacher_forcing)
    #     self.copied_inputs.append(copied_inputs)
    #     d_outputs_length = tf.count_nonzero(d_outputs_ph, axis=1, 
    #                                         name='outputs_length')
    #     with tf.name_scope('add_eos'):
    #       targets = tf.concat([d_outputs_ph, tf.zeros([batch_size, 1], dtype=tf.int32)], axis=1)

    #     # the length of outputs should be also added by 1 because of EOS. 
    #     with tf.name_scope('output_weights'):
    #       d_outputs_weights = tf.sequence_mask(
    #         d_outputs_length+1, maxlen=shape(d_outputs_ph, 1)+1, dtype=tf.float32)
    #     with tf.name_scope('loss%d' % i):
    #       loss = tf.contrib.seq2seq.sequence_loss(
    #         d_outputs, targets, d_outputs_weights)
    #   self.d_outputs_ph.append(d_outputs_ph)
    #   self.losses.append(loss)
    #   self.greedy_predictions.append(predictions)
    # with tf.name_scope('Loss'):
    #   self.loss = tf.reduce_mean(self.losses)
    # self.updates = self.get_updates(self.loss)

  def get_input_feed(self, batch, is_training):
    feed_dict = {
      self.e_inputs_w_ph: batch.w_sources,
      self.e_inputs_c_ph: batch.c_sources,
      self.d_outputs_ph: batch.targets,
      self.is_training: is_training,
    }
    return feed_dict

  def debug(self, data):
    pass

  def train(self, data):
    loss = 0.0
    num_steps = 0
    epoch_time = 0.0
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, True)
      t = time.time()
      step_loss, _ = self.sess.run([self.losses, self.updates], feed_dict)
      step_loss = np.mean(step_loss)
      epoch_time += time.time() - t
      loss += math.exp(step_loss)
      num_steps += 1
    loss /= num_steps
    return loss, epoch_time

  def test(self, data):
    inputs = []
    outputs = []
    predictions = []
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, False)
      predictions_dist = self.sess.run(self.predictions, feed_dict)
      batch_predictions = [np.argmax(dist, axis=2) for dist in predictions_dist]
      predictions.append(batch_predictions)
    return predictions

