# coding:utf-8

import sys, os, random, copy, collections, time, re, argparse
import pyhocon
import numpy as np
from pprint import pprint
from logging import FileHandler
import tensorflow as tf

from utils import common, evaluation, tf_utils
from core import models, datasets, vocabularies


tf_config = tf.ConfigProto(
  log_device_placement=False,
  allow_soft_placement=True, 
  gpu_options=tf.GPUOptions(
    allow_growth=True, # If False, all memories of the GPU will be occupied.
  )
)
default_config = common.recDotDict({
})

class Manager(object):
  @common.timewatch()
  def __init__(self, args, sess, vocab=None):
    self.sess = sess
    self.config = self.load_config(args)
    self.logger = common.logManager(handler=FileHandler(args.log_file)) if args.log_file else common.logManager()

    sys.stderr.write(str(self.config) + '\n')

    # self.w_vocab = WordVocabularyWithEmbedding(
    #   self.config.embeddings, 
    #   vocab_size=self.config.vocab_size, 
    #   lowercase=self.config.lowercase) if vocab is None else vocab
    data_class = getattr(datasets, self.config.dataset_type)
    if self.config.embeddings:
      emb_conf = self.config.embeddings
      self.w_vocab = vocabularies.WordVocabularyWithEmbedding(
        emb_conf.path, 
        vocab_size=self.config.w_vocab_size,
        skip_first=emb_conf.skip_first)
      self.c_vocab = None
    #self.w_vocab, self.c_vocab = data_class.create_vocab_from_data(self.config)
    self.dataset = data_class(self.config.dataset_info, 
                              self.w_vocab, self.c_vocab, 
                              context_max_len=self.config.context_max_len)

  def load_config(self, args):
    self.model_path = args.checkpoint_path
    self.summaries_path = self.model_path + '/summaries'
    self.checkpoints_path = self.model_path + '/checkpoints'
    self.tests_path = self.model_path + '/tests'
    self.config_path = args.config_path if args.config_path else self.model_path + '/config'

    # Read and restore config
    sys.stderr.write('Reading a config from %s ...\n' % (self.config_path))
    config = pyhocon.ConfigFactory.parse_file(self.config_path)
    config_restored_path = os.path.join(self.model_path, 'config')
    if not os.path.exists(self.summaries_path):
      os.makedirs(self.summaries_path)
    if not os.path.exists(self.checkpoints_path):
      os.makedirs(self.checkpoints_path)
    if not os.path.exists(self.tests_path):
      os.makedirs(self.tests_path)

    # Overwrite configs by temporary args. They have higher priorities than those in the config of models.
    if 'dataset_type' in args and args.dataset_type:
      config['dataset_type'] = args.dataset_type
    if 'train_data_size' in args and args.train_data_size:
      config['dataset_info']['train']['max_lines'] = args.train_data_size
    if 'train_data_path' in args and args.train_data_path:
      config['dataset_info']['train']['path'] = args.train_data_path
    if 'test_data_path' in args and args.test_data_path:
      config['dataset_info']['test']['path'] = args.test_data_path
    if 'batch_size' in args and args.batch_size:
      config['batch_size'] = args.batch_size
    if 'w_vocab_size' in args and args.w_vocab_size:
      config['w_vocab_size'] = args.w_vocab_size
    if 'target_attribute' in args and args.target_attribute:
      config['target_attribute'] = args.target_attribute

    if args.cleanup or not os.path.exists(config_restored_path):
      sys.stderr.write('Restore the config to %s ...\n' % (config_restored_path))

      with open(config_restored_path, 'w') as f:
        sys.stdout = f
        common.print_config(config)
        sys.stdout = sys.__stdout__
    config = common.recDotDict(config)
    default_config.update(config)
    config = default_config
    return config

  def save_model(self, model, save_as_best=False):
    checkpoint_path = self.checkpoints_path + '/model.ckpt'
    self.saver.save(self.sess, checkpoint_path, global_step=model.epoch)
    if save_as_best:
      suffixes = ['data-00000-of-00001', 'index', 'meta']
      for s in suffixes:
        source_path = self.checkpoints_path + "/model.ckpt-%d.%s" % (model.epoch.eval(), s)
        target_path = self.checkpoints_path + "/model.ckpt.best.%s" % (s)
        cmd = "cp %s %s" % (source_path, target_path)
        os.system(cmd)

  @common.timewatch()
  def train(self, model=None):
    if model is None:
      model = self.create_model(self.sess, self.config)
    testing_results = []
    for epoch in xrange(model.epoch.eval(), self.config.max_epoch):
      sys.stderr.write('Epoch %d:  Start Training...\n' % epoch)

      batches = self.dataset.train.get_batch(
        self.config.batch_size, 
        utterance_max_len=self.config.utterance_max_len, shuffle=True)
      loss, epoch_time = model.train(batches)
      
      summary = tf_utils.make_summary({
        'loss': loss
      })
      self.summary_writer.add_summary(summary, model.epoch.eval())

      self.logger.info('(Epoch %d) Train loss: %.3f (%.1f sec)' % (epoch, loss, epoch_time))
      
      # score = self.test(model=model, dataset=self.dataset.valid,
      #                   in_training=True)
      # if model.epoch.eval() == 0 or score > model.high_score.eval():
      #   save_as_best = True
      #   best_epoch = epoch
      #   model.update_highscore(score)
      #   self.logger.info('(Epoch %d) Update high score: %.3f' % (best_epoch, score))
      # else:
      #   save_as_best = False
      save_as_best = False
      self.save_model(model, save_as_best=save_as_best)
      model.add_epoch()
    return

  def debug(self):
    config = self.config
    batches = self.dataset.train.get_batch(
      config.batch_size, utterance_max_len=config.utterance_max_len,
      shuffle=False)
    #return batches
    self.dataset.train.load_data()
    print self.dataset.train.size
    for b in batches:
      for j, (w_context, c_context, response, speaker_change) in enumerate(zip(b.w_contexts, b.c_contexts, b.responses, b.speaker_changes)):
        print '<%d>' % j
        print 'w_contexts:',
        #print w_context
        print [self.w_vocab.id2sent(x, join=True) for x in w_context]
        if self.c_vocab:
          print 'c_contexts:', 
          #print c_context
          print [self.c_vocab.id2sent(x, join=True) for x in c_context]
          print 
        print 'response:',
        #print response
        print self.w_vocab.id2sent(response, join=True)
        print 'speaker_changes', speaker_change
      #exit(1)
    pass

  def demo(self, model=None, inp=None):
    if model is None:
      model = self.create_model(
        self.sess, self.config,
        checkpoint_path=self.checkpoints_path + '/model.ckpt.best')

  def test(self, model=None, dataset=None, verbose=True, in_training=False):
    config = self.config
    if dataset is None:
      dataset = self.dataset.test

    _, test_filename = common.separate_path_and_filename(
      self.config.dataset_info.test.path)

    if model is None: 
      model = self.create_model(
        self.sess, self.config,
        checkpoint_path=self.checkpoints_path + '/model.ckpt.best')
    batches = dataset.get_batch(
      config.batch_size, utterance_max_len=0, shuffle=False)
    if not in_training:
      sys.stderr.write('Start Decoding...\n')
    res, epoch_time = model.test(batches)
    test_filename = '%s.%02d' % (test_filename, model.epoch.eval()) if in_training else '%s.best' % (test_filename)
    test_output_path = os.path.join(self.tests_path, test_filename)

    with open(test_output_path, 'w') as f:
      sys.stdout = f
      for i, (context, response, prediction) in enumerate(zip(*res)):
        for j, c in enumerate(context):
          print '<%d-C%d>:\t%s' % (i, j, c)
        print '<%d-R>:\t%s' % (i, response)
        for j, p in enumerate(prediction):
          print '<%d-P%d>:\t%s' % (i, j, p)
        print ''
      sys.stdout = sys.__stdout__
    if in_training:
      self.summary_writer.add_summary(summary, model.epoch.eval())
    #return df

  @common.timewatch()
  def create_model(self, sess, config,
                   checkpoint_path=None, cleanup=False):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      m = getattr(models, config.model_type)(sess, config, self.w_vocab, self.c_vocab)

    if not checkpoint_path and not cleanup:
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None

    self.saver = tf.train.Saver(tf.global_variables(), config.max_to_keep)
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
      sys.stderr.write("Reading model parameters from %s\n" % checkpoint_path)
      self.saver.restore(sess, checkpoint_path)
    else:
      sys.stderr.write("Created model with fresh parameters.\n")
      sess.run(tf.global_variables_initializer())

    variables_path = self.model_path + '/variables.list'
    with open(variables_path, 'w') as f:
      variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
      f.write('\n'.join(variable_names) + '\n')

    self.summary_writer = tf.summary.FileWriter(self.summaries_path, 
                                                sess.graph)
    return m


def main(args):
  random.seed(0)
  np.random.seed(0)
  with tf.Graph().as_default(), tf.Session(config=tf_config).as_default() as sess:
    tf.set_random_seed(0)
    manager = Manager(args, sess)
    if args.mode == 'train':
      manager.train()
    elif args.mode == 'test':
      manager.test()
    elif args.mode == 'demo':
      manager.demo()
    elif args.mode == 'debug':
      manager.debug()
    else:
      raise ValueError('args.mode must be \'train\', \'test\', or \'demo\'.')

  if args.mode == 'train':
    vocab = manager.vocab
    with tf.Graph().as_default(), tf.Session(config=tf_config).as_default() as sess:
      tf.set_random_seed(0)
      manager = Manager(args, sess, vocab=vocab)
      manager.test()
  return manager

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint_path")
  parser.add_argument("mode")
  parser.add_argument("config_path")

  parser.add_argument("--cleanup", default=False, type=common.str2bool)
  parser.add_argument("--interactive", default=False, type=common.str2bool)
  parser.add_argument("--log_file", default=None, type=str)
  parser.add_argument("--train_data_size", default=None, type=int)
  parser.add_argument("--train_data_path", default=None, type=str)
  parser.add_argument("--test_data_path", default=None, type=str)
  parser.add_argument("--batch_size", default=None, type=int)
  parser.add_argument("--w_vocab_size", default=None, type=int)
  args  = parser.parse_args()
  main(args)

