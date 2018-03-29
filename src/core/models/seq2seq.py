# coding:utf-8
# https://github.com/tensorflow/tensorflow/issues/11598
import math, sys, time
import numpy as np
from pprint import pprint

import tensorflow as tf
from utils.tf_utils import shape
from utils.common import flatten, timewatch
from core.models.base import ModelBase, setup_cell
from core.models.encoder import CharEncoder, WordEncoder, RNNEncoder, CNNEncoder
from core.models import encoder
from core.extensions.pointer import pointer_decoder 
from core.vocabularies import BOS_ID, PAD_ID

# class SharedDenseLayer(tf.layers.Dense):
#   def __init__(self, *args, **kwargs):
#     super(SharedDenseLayer, self).__init__(*args, **kwargs)
#     self.kernel = None

#   def add_kernel(self, tensor):
#     assert shape(tensor, 0) == self.units
#     self.kernel = tensor 

#   def build(self, input_shape):
#     input_shape = tensor_shape.TensorShape(input_shape)
#     if input_shape[-1].value is None:
#       raise ValueError('The last dimension of the inputs to `Dense` '
#                        'should be defined. Found `None`.')
#     self.input_spec = base.InputSpec(min_ndim=2,
#                                      axes={-1: input_shape[-1].value})
#     if not self.kernel:
#       self.kernel = self.add_variable('kernel',
#                                       shape=[input_shape[-1].value, self.units],
#                                       initializer=self.kernel_initializer,
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint,
#                                       dtype=self.dtype,
#                                       trainable=True)
#     if self.use_bias:
#       self.bias = self.add_variable('bias',
#                                     shape=[self.units,],
#                                     initializer=self.bias_initializer,
#                                     regularizer=self.bias_regularizer,
#                                     constraint=self.bias_constraint,
#                                     dtype=self.dtype,
#                                     trainable=True)
#     else:
#       self.bias = None
#     self.built = True

class Seq2Seq(ModelBase):
  def __init__(self, sess, config, w_vocab, c_vocab):
    ModelBase.__init__(self, sess, config)
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.is_training = tf.placeholder(tf.bool, [], name='is_training')
    with tf.name_scope('keep_prob'):
      self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate

    # <Sample input>
    # e_inputs: [1, 40, 44, 0, 0], d_outputs: [2, 0, 0] (target=44)
    with tf.name_scope('Placeholders'):
      self.e_inputs_w_ph = tf.placeholder(
        tf.int32, [None, None, None], name="EncoderInputWords")

      # batch_size, context_len, utterance_len, word_len
      self.e_inputs_c_ph = tf.placeholder(
        tf.int32, [None, None, None, None], name="EncoderInputChars")

      # batch_size, utterance_len, 
      self.d_outputs_ph = tf.placeholder(
        tf.int32, [None, None], name="DecoderOutput")
        #tf.int32, [None, config.utterance_max_len], name="DecoderOutput")
      self.speaker_changes_ph = tf.placeholder(
        tf.int32, [None, None], name="SpeakerChanges")

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

      if c_vocab:
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
      sc_embeddings = self.initialize_embeddings(
          'SpeakerChange', [2, config.feature_size],
          trainable=trainable)
      speaker_changes = tf.nn.embedding_lookup(sc_embeddings, self.speaker_changes_ph)

    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
      assert self.w_vocab or self.c_vocab
      word_repls = []

      # Count the length of each dialogue, utterance, (word).
      uttr_lengths = tf.count_nonzero(self.e_inputs_w_ph, axis=2, dtype=tf.int32)
      dial_lengths = tf.count_nonzero(uttr_lengths, axis=1, dtype=tf.int32)

      with tf.variable_scope('Word') as scope:
        w_inputs = tf.nn.embedding_lookup(w_embeddings, self.e_inputs_w_ph)
        word_encoder = WordEncoder(self.keep_prob, shared_scope=scope)
        word_repls.append(word_encoder.encode(w_inputs))

      with tf.variable_scope('Char') as scope:
        if self.c_vocab:
          c_inputs = tf.nn.embedding_lookup(c_embeddings, self.e_inputs_c_ph)
          char_encoder = CNNEncoder(self.keep_prob, shared_scope=scope)
          word_repls.append(char_encoder.encode(c_inputs))

      self.word_repls = word_repls = tf.concat(word_repls, axis=-1) # [batch_size, context_len, utterance_len, word_emb_size + cnn_output_size]


      with tf.variable_scope('Utterance') as scope:
        uttr_encoder_type = getattr(encoder, config.uttr_encoder_type)
        uttr_encoder = uttr_encoder_type(config, self.keep_prob, 
                                         shared_scope=scope)
        print word_repls
        uttr_repls, _ = uttr_encoder.encode(word_repls, uttr_lengths)
        print uttr_repls
        exit(1)

        # Concatenate the feature_embeddings with each utterance representations.
        uttr_repls = tf.concat([uttr_repls, speaker_changes], axis=-1)

      with tf.variable_scope('Dialogue') as scope:
        dial_encoder_type = getattr(encoder, config.uttr_encoder_type)
        dial_encoder = dial_encoder_type(config, self.keep_prob, 
                                         shared_scope=scope)
        encoder_outputs, encoder_state = dial_encoder.encode(
          utter_repls, dial_lengths)

    ## Decoder
    with tf.variable_scope('Decoder') as scope:
      '''
      When a text such as ['how', 'are', 'you', '?'] is given to the decoder's placeholder,
       - decoder's input : ['_BOS', 'how', 'are', 'you', '?']
       - decoder's output (target) : ['how', 'are', 'you', '?', '_PAD']
      Here, the token _PAD behaves as a EOS.
      '''
      with tf.name_scope('start_tokens'):
        start_tokens = tf.tile(tf.constant([BOS_ID], dtype=tf.int32), [batch_size])
      with tf.name_scope('end_tokens'):
        end_token = PAD_ID
        end_tokens = tf.tile(tf.constant([end_token], dtype=tf.int32), [batch_size])
      with tf.name_scope('decoder_inputs'):
        decoder_inputs = tf.concat([tf.expand_dims(start_tokens, 1), self.d_outputs_ph], axis=1)
      # the length of decoder's inputs/outputs is increased by 1 because of BOS or EOS.
      with tf.name_scope('target_lengths'):
        target_length = tf.count_nonzero(self.d_outputs_ph, axis=1, dtype=tf.int32)+1 
      with tf.name_scope('target_weights'):
        target_weights = tf.sequence_mask(target_length, dtype=tf.float32)

      with tf.name_scope('targets'):
        targets = tf.concat([self.d_outputs_ph, tf.expand_dims(end_tokens, 1)], axis=1)[:, :shape(target_weights, 1)]


      decoder_inputs_emb = tf.nn.embedding_lookup(w_embeddings, decoder_inputs)
      helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_inputs_emb, sequence_length=target_length, time_major=False)
      
      # TODO: 多言語対応にする時はbias, trainableをfalseにしてembeddingをconstantにしたい

      decoder_cell = setup_cell(config.cell_type, config.rnn_size, 
                                config.num_layers,keep_prob=self.keep_prob)
      projection_layer = tf.layers.Dense(config.w_vocab_size,
                                         use_bias=True, trainable=True)

      encoder_input_length = dial_lengths
      attention_states = encoder_outputs
      num_units = shape(attention_states, -1)
      with tf.name_scope('Training'):
        attention = tf.contrib.seq2seq.LuongAttention(
          num_units, attention_states,
          memory_sequence_length=encoder_input_length)
        train_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
          decoder_cell, attention)

        # encoder_state can't be directly copied into decoder_cell when using the attention mechanisms, initial_state must be an instance of AttentionWrapperState. (https://github.com/tensorflow/nmt/issues/205)
        decoder_initial_state = train_decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
        decoder = tf.contrib.seq2seq.BasicDecoder(
          train_decoder_cell, helper, decoder_initial_state,
          output_layer=projection_layer)
          
        train_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
          decoder, 
          impute_finished=True,
          maximum_iterations=tf.reduce_max(target_length),
          scope=scope)
        logits = train_decoder_outputs.rnn_output

      with tf.name_scope('Test'):
        beam_width = config.beam_width
        attention = tf.contrib.seq2seq.LuongAttention(
          num_units, 
          tf.contrib.seq2seq.tile_batch(
            attention_states, multiplier=beam_width),
          memory_sequence_length=tf.contrib.seq2seq.tile_batch(
            encoder_input_length, multiplier=beam_width))
        test_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
          decoder_cell, attention)

        decoder_initial_state = test_decoder_cell.zero_state(batch_size*beam_width, tf.float32).clone(cell_state=tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width))
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
          test_decoder_cell, w_embeddings, start_tokens, end_token, 
          decoder_initial_state,
          beam_width, output_layer=projection_layer,
          length_penalty_weight=config.length_penalty_weight)
        test_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
          decoder, impute_finished=False,
          maximum_iterations=config.utterance_max_len, scope=scope)
        self.predictions = test_decoder_outputs.predicted_ids
          #FinalBeamDecoderOutput(predicted_ids=<tf.Tensor 'Decoder/Decode/Test/Decode/transpose:0' shape=(?, ?, 3) dtype=int32>, beam_search_decoder_output=BeamSearchDecoderOutput(scores=<tf.Tensor 'Decoder/Decode/Test/Decode/transpose_1:0' shape=(?, ?, 3) dtype=float32>, predicted_ids=<tf.Tensor 'Decoder/Decode/Test/Decode/transpose_2:0' shape=(?, ?, 3) dtype=int32>, parent_ids=<tf.Tensor 'Decoder/Decode/Test/Decode/transpose_3:0' shape=(?, ?, 3) dtype=int32>))

    with tf.name_scope('Loss'):
      self.loss = tf.contrib.seq2seq.sequence_loss(
        logits, targets, target_weights,
        average_across_timesteps=True, average_across_batch=True)
    self.updates = self.get_updates(self.loss)
    self.debug = []
    self.debug = [self.e_inputs_w_ph, self.d_outputs_ph, decoder_inputs, targets, target_length, target_weights]

  def get_input_feed(self, batch, is_training):
    feed_dict = {
      self.d_outputs_ph: np.array(batch.responses),
      self.is_training: is_training,
      self.speaker_changes_ph: np.array(batch.speaker_changes)
    }
    feed_dict[self.e_inputs_w_ph] = np.array(batch.w_contexts)
    if self.c_vocab:
      feed_dict[self.e_inputs_c_ph] = np.array(batch.c_contexts)
    # print '<<<<<get_input_feed>>>>'
    # for k,v in feed_dict.items():
    #   if type(v) == np.ndarray:
    #     print k, v #v.shape

    return feed_dict

  @timewatch()
  def train(self, data, do_update=True):
    '''
    This method can be used for the calculation of valid loss with do_update=False
    '''
    loss = 0.0
    num_steps = 0
    epoch_time = 0.0
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, do_update)
      # for x,resx in zip(self.debug, self.sess.run(self.debug, feed_dict)):
      #   print x
      #   print resx, resx.shape
      # exit(1)
      t = time.time()
      output_feed = [self.loss, self.updates] if do_update else [self.loss]
      res = self.sess.run(output_feed, feed_dict)
      step_loss = math.exp(res[0])
      print self.epoch.eval(), i, step_loss
      sys.stdout.flush()
      if math.isnan(step_loss):
        sys.stderr.write('Got a Nan loss.\n')
        for x in feed_dict:
          print x
          print feed_dict[x]
        exit(1)
      epoch_time += time.time() - t
      loss += step_loss
      num_steps += 1
    loss /= num_steps
    return loss, epoch_time

  @timewatch()
  def test(self, data):
    inputs = []
    outputs = []
    speaker_changes = []
    predictions = []
    num_steps = 0
    epoch_time = 0.0
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, False)
      # for x,resx in zip(self.debug, self.sess.run(self.debug, feed_dict)):
      #    print x
      #    print resx.shape
      # exit(1)
      t = time.time()
      batch_predictions = self.sess.run(self.predictions, feed_dict)
      epoch_time += time.time() - t
      num_steps += 1
      inputs.append(batch.w_contexts)
      outputs.append(batch.responses)
      speaker_changes.append(batch.speaker_changes)
      predictions.append(batch_predictions)
    inputs = flatten(inputs)
    outputs = flatten(outputs)
    speaker_changes = flatten(speaker_changes)
    predictions = flatten(predictions)
    inputs = [[self.w_vocab.id2sent(u, join=True) for u in c] for c in inputs]
    outputs = [self.w_vocab.id2sent(r, join=True) for r in outputs]
    # [batch_size, utterance_max_len, beam_width] - > [batch_size, beam_width, utterance_max_len]
    predictions = [[self.w_vocab.id2sent(r, join=True) for r in zip(*p)] for p in predictions] 
    return (inputs, outputs, speaker_changes, predictions), epoch_time

