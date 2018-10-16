import numpy as np
import keras
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, Activation, Embedding, Input, concatenate, add, Bidirectional, LSTM
from keras import Model, regularizers
from keras.optimizers import Adam, Adadelta, RMSprop, Adagrad

import os
import sys
import logging

from metrics import recall, precision, f1, calculate_metrics
from load_data_ddi import load_word_matrix
from seq_self_attention import SeqSelfAttention

# Make the directory for saving model, weight, log
result_dir = 'result'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


class CNN(object):
    def __init__(self,
                 max_sent_len,
                 vocb,
                 num_classes,
                 emb_dim=100,
                 pos_dim=10,
                 kernel_lst=[3, 4, 5],
                 nb_filters=100,
                 dropout_rate=0.2,
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
        self.dropout_rate = dropout_rate
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
        self.compile_model()

    def add_input_layer(self):
        self.input_x = Input(shape=(self.max_sent_len,), dtype='int32')
        self.input_pos = Input(shape=(self.max_sent_len,), dtype='int32')

    def add_embedding_layer(self):
        # If static, trainable = False. If non-static, trainable = True
        if self.use_pretrained:
            # load word matrix
            word_matrix = load_word_matrix(self.vocb, self.emb_dim, self.unk_limit)
            # If static, trainable = False. If non-static, trainable = True
            self.w_emb = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim, input_length=self.max_sent_len,
                                   trainable=self.non_static, weights=[word_matrix])(self.input_x)
        else:
            # If static, trainable = False. If non-static, trainable = True
            self.w_emb = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim,
                                   input_length=self.max_sent_len, trainable=self.non_static)(self.input_x)

        # Position Embedding (0, 1, 2)
        self.pos_emb = Embedding(input_dim=3, output_dim=self.pos_dim, input_length=self.max_sent_len, trainable=True)(self.input_pos)

        # Concatenation
        self.emb_concat = concatenate([self.w_emb, self.pos_emb])

    def add_cnn_layer(self):
        # CNN Parts
        layer_lst = []
        for kernel_size in self.kernel_lst:
            conv_l = Conv1D(filters=self.nb_filters, kernel_size=kernel_size, padding='valid', activation='relu')(self.emb_concat)
            pool_l = MaxPool1D(pool_size=self.max_sent_len - kernel_size + 1)(conv_l)
            drop_l = Dropout(self.dropout_rate)(pool_l)
            # Append the final result
            layer_lst.append(drop_l)

        self.concat_l = concatenate(layer_lst)

    def add_fc_layer(self):
        self.concat_drop_l = Dropout(self.dropout_rate)(self.concat_l)
        self.flat_l = Flatten()(self.concat_drop_l)
        self.pred_output = Dense(self.num_classes, activation='softmax')(self.flat_l)

    def compile_model(self):
        self.model = Model(inputs=[self.input_x, self.input_pos], outputs=self.pred_output)
        # Optimizer
        if self.optimizer.lower() == 'adam':
            opt = Adam(lr=self.lr_rate)
        elif self.optimizer.lower() == 'rmsprop':
            opt = RMSprop(lr=self.lr_rate)
        elif self.optimizer.lower() == 'adagrad':
            opt = Adagrad(lr=self.lr_rate)
        elif self.optimizer.lower() == 'adadelta':
            opt = Adadelta(lr=self.lr_rate)
        else:
            raise ValueError("Use Optimizer in Adam, RMSProp, Adagrad, Adadelta!")
        # Model compile
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy', precision, recall, f1])

    def save_model(self):
        # Save the model into the result directory
        model_json = self.model.to_json()
        with open(os.path.join(result_dir, 'model.json'), "w", encoding='utf-8') as json_file:
            json_file.write(model_json)
        print('Save model.json')

    def train(self, sentence, pos_lst, y, nb_epoch, batch_size, validation_split):
        # Write log
        logging.basicConfig(filename=os.path.join(result_dir, 'result.log'),
                            level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', filemode='a')
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler(sys.stdout))  # For print out the result on console
        logger.info('')
        logger.info("#################################### New Start #####################################")

        max_val_f1 = 0
        for i in range(nb_epoch):
            # Training
            train_history = self.model.fit(x=[sentence, pos_lst], y=y, epochs=1, batch_size=batch_size, verbose=1, validation_split=validation_split)
            # Writing the log
            train_loss = train_history.history['loss'][0]
            train_acc = train_history.history['acc'][0]
            train_f1 = train_history.history['f1'][0]
            train_p = train_history.history['precision'][0]
            train_r = train_history.history['recall'][0]

            val_loss = train_history.history['val_loss'][0]
            val_acc = train_history.history['val_acc'][0]
            val_f1 = train_history.history['val_f1'][0]
            val_p = train_history.history['val_precision'][0]
            val_r = train_history.history['val_recall'][0]
            logger.info('##train##, epoch: {:2d}, loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format((i + 1),
                                                                                                                              train_loss,
                                                                                                                              train_acc,
                                                                                                                              train_p,
                                                                                                                              train_r,
                                                                                                                              train_f1))
            logger.info('##dev##,   epoch: {:2d}, loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format((i+1),
                                                                                                                              val_loss,
                                                                                                                              val_acc,
                                                                                                                              val_p,
                                                                                                                              val_r,
                                                                                                                              val_f1))
            # Saving the weight if it is better than before (early-stopping)
            if max_val_f1 < val_f1:
                max_val_f1 = val_f1
                logging.info("[{}th epoch, Better performance! Update the weight!]".format(i + 1))
                self.model.save_weights(os.path.join(result_dir, 'weights.h5'))

    def evaluate(self, sentence, pos_lst, y, batch_size):
        self.model.load_weights(os.path.join(result_dir, 'weights.h5'))
        pred_test = self.model.evaluate(x=[sentence, pos_lst], y=y, batch_size=batch_size, verbose=1)
        logger = logging.getLogger()
        logger.info(
            '##test##,  loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(pred_test[0],
                                                                                                    pred_test[1],
                                                                                                    pred_test[2],
                                                                                                    pred_test[3],
                                                                                                    pred_test[4],))

    def show_model_summary(self):
        print(self.model.summary(line_length=100))


class MCCNN(CNN):
    def __init__(self,
                 max_sent_len,
                 vocb,
                 num_classes,
                 emb_dim=100,
                 pos_dim=10,
                 kernel_lst=[3, 4, 5],
                 nb_filters=100,
                 dropout_rate=0.2,
                 optimizer='adam',
                 lr_rate=0.001,
                 use_pretrained=False,
                 unk_limit=10000):
        self.max_sent_len = max_sent_len
        self.vocb = vocb
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.pos_dim = pos_dim
        self.kernel_lst = kernel_lst
        self.nb_filters = nb_filters
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.lr_rate = lr_rate
        self.use_pretrained = use_pretrained
        self.unk_limit = unk_limit
        self.build_model()

    def add_embedding_layer(self):
        # If static, trainable = False. If non-static, trainable = True
        if self.use_pretrained:
            # load word matrix
            word_matrix = load_word_matrix(self.vocb, self.emb_dim, self.unk_limit)
            # If static, trainable = False. If non-static, trainable = True
            self.w_emb_static = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim, input_length=self.max_sent_len,
                                          trainable=False, weights=[word_matrix])(self.input_x)
            self.w_emb_non_static = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim, input_length=self.max_sent_len,
                                              trainable=True, weights=[word_matrix])(self.input_x)
        else:
            # If static, trainable = False. If non-static, trainable = True
            self.w_emb_static = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim,
                                          input_length=self.max_sent_len, trainable=False)(self.input_x)
            self.w_emb_non_static = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim,
                                              input_length=self.max_sent_len, trainable=True)(self.input_x)
        # Position Embedding (0, 1, 2)
        self.pos_emb = Embedding(input_dim=3, output_dim=self.pos_dim, input_length=self.max_sent_len, trainable=True)(self.input_pos)

        # Concatenation
        self.w_emb_static_concat = concatenate([self.w_emb_static, self.pos_emb])
        self.w_emb_non_static_concat = concatenate([self.w_emb_non_static, self.pos_emb])

    def add_cnn_layer(self):
        # CNN Parts
        layer_lst = []
        for kernel_size in self.kernel_lst:
            # Sharing the filter weight
            conv_layer = Conv1D(filters=self.nb_filters, kernel_size=kernel_size, padding='valid', activation='relu')
            pool_layer = MaxPool1D(pool_size=self.max_sent_len - kernel_size + 1)
            # Static layer
            conv_static = conv_layer(self.w_emb_static_concat)
            pool_static = pool_layer(conv_static)
            # Non-static layer
            conv_non_static = conv_layer(self.w_emb_non_static_concat)
            pool_non_static = pool_layer(conv_non_static)
            # Add two layer
            add_l = add([pool_static, pool_non_static])
            drop_l = Dropout(self.dropout_rate)(add_l)
            # Append the final result
            layer_lst.append(drop_l)

        self.concat_l = concatenate(layer_lst)


class BILSTM(CNN):
    def __init__(self,
                 max_sent_len,
                 vocb,
                 num_classes,
                 emb_dim=100,
                 pos_dim=10,
                 rnn_dim=100,
                 dropout_rate=0.2,
                 optimizer='adam',
                 lr_rate=0.001,
                 non_static=True,
                 use_pretrained=False,
                 unk_limit=10000,
                 use_self_att=False):
        self.max_sent_len = max_sent_len
        self.vocb = vocb
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.pos_dim = pos_dim
        self.rnn_dim = rnn_dim
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.lr_rate = lr_rate
        self.non_static = non_static
        self.use_pretrained = use_pretrained
        self.unk_limit = unk_limit
        self.use_self_att = use_self_att
        self.build_model()

    def build_model(self):
        self.add_input_layer()
        self.add_embedding_layer()
        self.add_rnn_layer()
        if self.use_self_att:
            self.add_self_att_layer()
        else:
            self.add_fc_layer()
        self.compile_model()

    def add_rnn_layer(self):
        self.rnn_l = Bidirectional(LSTM(self.rnn_dim, dropout=self.dropout_rate,
                                        recurrent_dropout=self.dropout_rate, return_sequences=True))(self.emb_concat)

    def add_self_att_layer(self):
        self.att_l = SeqSelfAttention(attention_activation='sigmoid')(self.rnn_l)
        # self.att_l = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        #                         kernel_regularizer=keras.regularizers.l2(1e-4),
        #                         bias_regularizer=keras.regularizers.l1(1e-4),
        #                         attention_regularizer_weight=1e-4,
        #                         name='Attention')(self.rnn_l)
        self.flat_l = Flatten()(self.att_l)
        self.dense_1000_l = Dense(1000)(self.flat_l)
        self.pred_output = Dense(self.num_classes, activation='softmax')(self.dense_1000_l)

    def add_fc_layer(self):
        self.flat_l = Flatten()(self.rnn_l)
        self.pred_output = Dense(self.num_classes, activation='softmax')(self.flat_l)
