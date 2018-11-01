import tensorflow as tf
import numpy as np
import keras

import os
import sys
import logging
import random

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from load_data_ddi import load_word_matrix

# Make the directory for saving model, weight, log
result_dir = 'result'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


def batch_loader(iterable, batch_size, shuffle=False):
    length = len(iterable)
    if shuffle:
        random.shuffle(iterable)
    for start_idx in range(0, length, batch_size):
        yield iterable[start_idx: min(length, start_idx + batch_size)]


class CNN(object):
    def __init__(self,
                 max_sent_len,
                 vocb,
                 num_classes,
                 emb_dim=100,
                 pos_dim=10,
                 kernel_lst=[3, 4, 5],
                 nb_filters=100,
                 optimizer='adam',
                 lr_rate=0.001,
                 non_static=True,
                 use_pretrained=False,
                 unk_limit=10000):
        self.max_sent_len = max_sent_len
        self.vocb = vocb
        self.emb_dim = emb_dim
        self.pos_dim = pos_dim
        self.kernel_lst = kernel_lst
        self.nb_filters = nb_filters
        self.optimizer = optimizer
        self.lr_rate = lr_rate
        self.non_static = non_static
        self.use_pretrained = use_pretrained
        self.unk_limit = unk_limit
        self.num_classes = num_classes
        self.build_model()

    def build_model(self):
        self.add_input_layer()
        self.add_embedding_layer()
        self.add_cnn_layer()
        self.add_fc_layer()

    def add_input_layer(self):
        self.input_x = tf.placeholder(shape=[None, self.max_sent_len], dtype=tf.int32, name='input_x')
        self.input_pos = tf.placeholder(shape=[None, self.max_sent_len], dtype=tf.int32, name='input_pos')
        self.input_y = tf.placeholder(shape=[None, self.num_classes], dtype=tf.int32, name='input_y')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

    def add_embedding_layer(self):
        '''
        TODO: loading pretrained not implemented
        '''
        # Word embedding
        self.w_emb = tf.get_variable(
            shape=[len(self.vocb), self.emb_dim],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(), name='embedding_layer')
        # Position embedding (0, 1, 2)
        self.pos_emb = tf.get_variable(
            shape=[3, self.pos_dim],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(), name='position_layer')
        # Embedding lookup
        self.word_lookup = tf.nn.embedding_lookup(self.w_emb, self.input_x)
        self.pos_lookup = tf.nn.embedding_lookup(self.pos_emb, self.input_pos)

        # Concatenation
        self.emb_concat = tf.concat([self.word_lookup, self.pos_lookup], axis=2)
        # Expanding dimenstion
        self.emb_concat_expand = tf.expand_dims(self.emb_concat, axis=3)

    def add_cnn_layer(self):
        pooled_output = []
        for i, kernel_size in enumerate(self.kernel_lst):
            # TODO: emb_dim + pos_dim = 310
            filter_shape = [kernel_size, self.emb_dim + self.pos_dim, 1, self.nb_filters]
            W = tf.get_variable(shape=filter_shape,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                name="filter_{}".format(i),
                                dtype=tf.float32)
            b = tf.get_variable(shape=self.nb_filters,
                                initializer=tf.zeros_initializer(),
                                name="bias_{}".format(i),
                                dtype=tf.float32)
            conv_l = tf.nn.conv2d(
                self.emb_concat_expand,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="conv_{}".format(i)
            )
            relu_l = tf.nn.relu(tf.nn.bias_add(conv_l, b), name='relu_{}'.format(i))
            pool_l = tf.nn.max_pool(
                relu_l,
                ksize=[1, self.max_sent_len - kernel_size + 1, 1, 1],
                padding='VALID',
                strides=[1, 1, 1, 1]
            )
            pooled_output.append(pool_l)
        # Concat
        self.feature_maps = tf.concat(pooled_output, axis=3)
        # Dropout
        self.feature_maps_drop = tf.nn.dropout(self.feature_maps, self.dropout_keep_prob)
        # Flatten
        self.feature_maps_flat = tf.reshape(self.feature_maps_drop,
                                            [-1, self.nb_filters * len(self.kernel_lst)])

    def add_fc_layer(self):
        self.logits = tf.contrib.layers.fully_connected(self.feature_maps_flat,
                                                        self.num_classes,
                                                        activation_fn=None)
        self.preds = tf.argmax(self.logits, axis=1)
        correct_predictions = tf.equal(self.preds, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32), name='accuracy')
        # self.f1 = tf.contrib.metrics.f1_score(self.input_y, self.preds)
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.input_y)
        self.loss = tf.reduce_mean(losses)
        opt = tf.train.AdamOptimizer(self.lr_rate)
        self.global_step = tf.Variable(tf.constant(0), trainable=False, name="global_step")
        self.train_op = opt.minimize(self.loss, global_step=self.global_step)

    def save_model(self):
        pass

    def train(self, sess, sentence, pos_lst, y, dropout_keep_prob):
        input_feed = {
            self.input_x: sentence,
            self.input_pos: pos_lst,
            self.input_y: y,
            self.dropout_keep_prob: dropout_keep_prob
        }
        output_feed = [self.train_op, self.global_step, self.loss, self.accuracy]
        outputs = sess.run(output_feed, feed_dict=input_feed)
        return outputs[1], outputs[2], outputs[3]

    def evaluate(self, sess, sentence, pos_lst, y):
        input_feed = {
            self.input_x: sentence,
            self.input_pos: pos_lst,
            self.input_y: y,
            self.dropout_keep_prob: 1.0
        }
        output_feed = [self.loss, self.accuracy, self.f1]
        outputs = sess.run(output_feed, feed_dict=input_feed)
        return outputs[0], outputs[1], outputs[2]

    def show_model_summary(self):
        pass
