import numpy as np
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, Embedding, Input, concatenate, add, Bidirectional, LSTM, BatchNormalization, \
    GlobalMaxPool1D, Activation, SeparableConv1D, Reshape, Conv2D, GlobalMaxPool2D, ZeroPadding2D
from keras import Model
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam, Adadelta, RMSprop, Adagrad
from sklearn.metrics import f1_score, recall_score, precision_score

# AutoML library
# from hyperopt import Trials, STATUS_OK, tpe
# from hyperas import optim
# from hyperas.distributions import uniform, choice

import os
import sys
import logging

from load_data_ddi import load_test_pair_id, load_word_matrix_all, load_word_matrix_from_txt
from seq_self_attention import SeqSelfAttention
from utils import save_best_result

# Make the directory for saving model, weight, log
result_dir = 'result'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# Make tensorboard log directory
tf_board_dir = 'tf_board_log'
if not os.path.exists(tf_board_dir):
    os.mkdir(tf_board_dir)


class CNN(object):
    def __init__(self, cfg, vocb, d1_vocb, d2_vocb):
        self.cfg = cfg
        self.vocb = vocb
        self.d1_vocb = d1_vocb
        self.d2_vocb = d2_vocb
        self.nb_epoch = cfg.nb_epoch
        self.batch_size = cfg.batch_size
        self.max_sent_len = cfg.max_sent_len
        self.emb_dim = cfg.emb_dim
        self.pos_dim = cfg.pos_dim
        self.kernel_lst = cfg.kernel_lst
        self.nb_filters = cfg.nb_filters
        self.dropout_rate = cfg.dropout_rate
        self.optimizer = cfg.optimizer
        self.lr_rate = cfg.lr_rate
        self.non_static = cfg.non_static
        self.use_pretrained = cfg.use_pretrained
        self.unk_limit = cfg.unk_limit
        self.num_classes = cfg.num_classes
        self.hidden_unit_size = cfg.hidden_unit_size
        self.use_batch_norm = cfg.use_batch_norm
        self.use_l2_reg = cfg.use_l2_reg
        self.reg_coef_conv = cfg.reg_coef_conv
        self.reg_coef_dense = cfg.reg_coef_dense
        self.build_model()

    def build_model(self):
        self.add_input_layer()
        self.add_embedding_layer()
        self.add_cnn_layer()
        self.add_fc_layer()
        self.compile_model()

    def add_input_layer(self):
        self.input_x = Input(shape=(self.max_sent_len,), dtype='int32')
        self.input_d1 = Input(shape=(self.max_sent_len,), dtype='int32')
        self.input_d2 = Input(shape=(self.max_sent_len,), dtype='int32')

    def add_embedding_layer(self):
        if self.unk_limit < len(self.vocb):
            input_dim_len = self.unk_limit
        else:
            input_dim_len = len(self.vocb)

        # If static, trainable = False. If non-static, trainable = True
        if self.use_pretrained:
            # load word matrix
            word_matrix = load_word_matrix_from_txt(self.vocb, self.emb_dim, self.unk_limit, 'pubmed_and_pmc')
            # If static, trainable = False. If non-static, trainable = True
            self.w_emb = Embedding(input_dim=input_dim_len, output_dim=self.emb_dim, input_length=self.max_sent_len,
                                   trainable=self.non_static, weights=[word_matrix])(self.input_x)
        else:
            # If static, trainable = False. If non-static, trainable = True

            self.w_emb = Embedding(input_dim=input_dim_len, output_dim=self.emb_dim,
                                   input_length=self.max_sent_len, trainable=self.non_static)(self.input_x)

        # Position Embedding
        # d1
        self.d1_emb = Embedding(input_dim=len(self.d1_vocb), output_dim=self.pos_dim, input_length=self.max_sent_len, trainable=True)(self.input_d1)
        # d2
        self.d2_emb = Embedding(input_dim=len(self.d2_vocb), output_dim=self.pos_dim, input_length=self.max_sent_len, trainable=True)(self.input_d2)
        # Concatenation
        self.emb_concat = concatenate([self.w_emb, self.d1_emb, self.d2_emb])

    def add_cnn_layer(self):
        # CNN Parts
        layer_lst = []
        for kernel_size in self.kernel_lst:
            if self.use_l2_reg:
                conv_l = Conv1D(filters=self.nb_filters, kernel_size=kernel_size, padding='valid', kernel_regularizer=l2(self.reg_coef_conv))(
                    self.emb_concat)
            else:
                conv_l = Conv1D(filters=self.nb_filters, kernel_size=kernel_size, padding='valid')(self.emb_concat)
            if self.use_batch_norm:
                conv_l = BatchNormalization()(conv_l)
            conv_l = Activation('relu')(conv_l)
            conv_l = GlobalMaxPool1D()(conv_l)
            # Append the final result
            layer_lst.append(conv_l)

        if len(layer_lst) != 1:
            self.concat_l = concatenate(layer_lst)
        else:
            self.concat_l = layer_lst[0]

        # Dropout
        self.concat_l = Dropout(self.dropout_rate)(self.concat_l)

    def add_fc_layer(self):
        self.fc_l = self.concat_l
        if self.use_l2_reg:
            self.fc_l = Dense(self.hidden_unit_size, kernel_regularizer=l2(self.reg_coef_dense))(self.fc_l)
        else:
            self.fc_l = Dense(self.hidden_unit_size)(self.fc_l)
        if self.use_batch_norm:
            self.fc_l = BatchNormalization()(self.fc_l)
        self.fc_l = Activation('relu')(self.fc_l)
        self.fc_l = Dropout(self.dropout_rate)(self.fc_l)
        if self.use_l2_reg:
            self.pred_output = Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(self.reg_coef_dense))(self.fc_l)
        else:
            self.pred_output = Dense(self.num_classes, activation='softmax')(self.fc_l)

    def compile_model(self):
        self.model = Model(inputs=[self.input_x, self.input_d1, self.input_d2], outputs=self.pred_output)
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
                           metrics=['accuracy'])

    def save_model(self):
        # Save the model into the result directory
        model_json = self.model.to_json()
        with open(os.path.join(result_dir, 'model.json'), "w", encoding='utf-8') as json_file:
            json_file.write(model_json)
        print('Save model.json')

    def write_hyperparam(self):
        # mode: cnn | nb_epoch: 30 | batch: 200 | opt: adam | lr: 0.007 | pretrain: True | k_lst: [3, 4, 5] | nb_filters: 100 |
        # emb_dim: 200 | pos_dim: 10 | sent_len: 150 | dropout: 0.8
        # Write k_lst
        k_lst_str = "["
        for kernel in self.kernel_lst:
            k_lst_str += "{} ".format(kernel)
        k_lst_str = k_lst_str.rstrip()
        k_lst_str += "]"
        log_str_1 = "mode: {} | nb_epoch: {} | batch: {} | opt: {} | lr: {} | pretrain: {} | hidden_unit_size: {}".format(type(self).__name__,
                                                                                                                          self.nb_epoch,
                                                                                                                          self.batch_size,
                                                                                                                          self.optimizer,
                                                                                                                          self.lr_rate,
                                                                                                                          self.use_pretrained,
                                                                                                                          self.hidden_unit_size)
        log_str_2 = "k_lst: {} | nb_filters: {} | emb_dim: {} | pos_dim: {} | sent_len: {} | dropout: {}".format(k_lst_str, self.nb_filters,
                                                                                                                 self.emb_dim, self.pos_dim,
                                                                                                                 self.max_sent_len, self.dropout_rate)
        return log_str_1, log_str_2

    def train(self, train_data, dev_data):
        # Unpack data
        (tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_y) = train_data
        (de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_y) = dev_data

        # Write log
        logging.basicConfig(filename=os.path.join(result_dir, 'result.log'),
                            level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', filemode='a')
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler(sys.stdout))  # For print out the result on console
        logger.info('')
        logger.info("#################################### New Start #####################################")
        # Write log for the hyperparameter
        log_str_1, log_str_2 = self.write_hyperparam()
        logger.info(log_str_1)
        logger.info(log_str_2)
        logger.info('')

        # Training
        self.model.fit(x=[tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst], y=tr_y, epochs=self.nb_epoch,
                       batch_size=self.batch_size,
                       verbose=1, validation_data=[[de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst], de_y],
                       callbacks=self.cfg.callback_list)

        # load the best result
        self.model.load_weights(os.path.join(result_dir, 'weights.h5'))

        # Metrics for Train data
        pred_tr = self.model.predict(x=[tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst], batch_size=self.batch_size, verbose=1)
        # train_loss = train_history.history['loss'][0]
        # train_acc = train_history.history['acc'][0]
        train_f1 = f1_score(np.argmax(tr_y, 1), np.argmax(pred_tr, 1), [1, 2, 3, 4], average='micro')
        train_p = precision_score(np.argmax(tr_y, 1), np.argmax(pred_tr, 1), [1, 2, 3, 4], average='micro')
        train_r = recall_score(np.argmax(tr_y, 1), np.argmax(pred_tr, 1), [1, 2, 3, 4], average='micro')

        # Metrics for Dev data
        pred_de = self.model.predict(x=[de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst], batch_size=self.batch_size, verbose=1)
        val_f1 = f1_score(np.argmax(de_y, 1), np.argmax(pred_de, 1), [1, 2, 3, 4], average='micro')
        val_p = precision_score(np.argmax(de_y, 1), np.argmax(pred_de, 1), [1, 2, 3, 4], average='micro')
        val_r = recall_score(np.argmax(de_y, 1), np.argmax(pred_de, 1), [1, 2, 3, 4], average='micro')

        # Writing the log
        logger.info('##train##, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(train_p, train_r, train_f1))
        logger.info('##dev##,   prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(val_p, val_r, val_f1))

    def evaluate(self, test_data):
        (sentences2idx, d1_lst, d2_lst), y = test_data
        self.model.load_weights(os.path.join(result_dir, 'weights.h5'))
        pred_te = self.model.predict(x=[sentences2idx, d1_lst, d2_lst], batch_size=self.batch_size, verbose=1)
        te_f1 = f1_score(np.argmax(y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
        te_p = precision_score(np.argmax(y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
        te_r = recall_score(np.argmax(y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
        logger = logging.getLogger()
        logger.info('##test##, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(te_p,
                                                                                te_r,
                                                                                te_f1))
        # Save the best result
        save_best_result(type(self).__name__, te_f1, result_dir)

    def show_model_summary(self):
        print(self.model.summary(line_length=100))

    def predict(self, sentences2idx, d1_lst, d2_lst, one_hot=True):
        self.model.load_weights(os.path.join(result_dir, 'weights.h5'))
        y_pred = self.model.predict(x=[sentences2idx, d1_lst, d2_lst], batch_size=self.batch_size)
        print(y_pred.shape)
        print(y_pred[:5])
        if one_hot:
            return self.one_hot_encoding(y_pred)
        else:
            return y_pred

    def one_hot_encoding(self, y_pred):
        # Argmax
        arg_maxed = np.argmax(y_pred, axis=1)
        # One Hot encoding
        one_hot = np.zeros((arg_maxed.size, self.num_classes))
        one_hot[np.arange(arg_maxed.size), arg_maxed] = 1
        print(one_hot.shape)
        print(one_hot[:5])
        return one_hot


class MCCNN(CNN):
    def __init__(self, cfg, vocb, d1_vocb, d2_vocb):
        self.cfg = cfg
        self.vocb = vocb
        self.d1_vocb = d1_vocb
        self.d2_vocb = d2_vocb
        self.max_sent_len = cfg.max_sent_len
        self.nb_epoch = cfg.nb_epoch
        self.batch_size = cfg.batch_size
        self.num_classes = cfg.num_classes
        self.emb_dim = cfg.emb_dim
        self.pos_dim = cfg.pos_dim
        self.kernel_lst = cfg.kernel_lst
        self.nb_filters = cfg.nb_filters
        self.dropout_rate = cfg.dropout_rate
        self.optimizer = cfg.optimizer
        self.lr_rate = cfg.lr_rate
        self.unk_limit = cfg.unk_limit
        self.hidden_unit_size = cfg.hidden_unit_size
        self.use_batch_norm = cfg.use_batch_norm
        self.use_l2_reg = cfg.use_l2_reg
        self.reg_coef_conv = cfg.reg_coef_conv
        self.reg_coef_dense = cfg.reg_coef_dense
        self.build_model()

    def add_embedding_layer(self):
        # If static, trainable = False. If non-static, trainable = True
        # load word matrix
        word_matrix_lst = load_word_matrix_all(self.vocb, self.emb_dim, self.unk_limit)
        emb_lst = []
        # Position Embedding
        # d1
        d1_emb = Embedding(input_dim=len(self.d1_vocb), output_dim=self.pos_dim, input_length=self.max_sent_len, trainable=True)(self.input_d1)
        # d2
        d2_emb = Embedding(input_dim=len(self.d2_vocb), output_dim=self.pos_dim, input_length=self.max_sent_len, trainable=True)(self.input_d2)

        for word_matrix in word_matrix_lst:
            w_emb = Embedding(input_dim=len(word_matrix), output_dim=self.emb_dim, input_length=self.max_sent_len,
                              trainable=True, weights=[word_matrix])(self.input_x)
            # Concatenation
            emb_concat = concatenate([w_emb, d1_emb, d2_emb])
            emb_concat = Reshape((self.max_sent_len, self.emb_dim + self.pos_dim * 2, 1))(emb_concat)
            emb_lst.append(emb_concat)
        # Concat all the embeddings
        self.emb_concat = concatenate(emb_lst)

    def add_cnn_layer(self):
        # CNN Parts
        layer_lst = []
        for kernel_size in self.kernel_lst:
            # Sharing the filter weight
            if self.use_l2_reg:
                conv_layer = Conv2D(self.nb_filters, kernel_size=(kernel_size, self.emb_dim + self.pos_dim * 2), padding='same',
                                    strides=(1, self.emb_dim + self.pos_dim * 2),
                                    kernel_regularizer=l2(self.reg_coef_conv))
            else:
                conv_layer = Conv2D(self.nb_filters, kernel_size=(kernel_size, self.emb_dim + self.pos_dim * 2), padding='same',
                                    strides=(1, self.emb_dim + self.pos_dim * 2))
            # zero padding on embedding layer, only on height
            # if kernel_size % 2 == 0:
            #     padding_size = int(kernel_size / 2)
            # else:
            #     padding_size = int((kernel_size - 1) / 2)
            # padded_emb_concat = ZeroPadding2D((padding_size, 0))(self.emb_concat)

            # Convolution
            conv_l = conv_layer(self.emb_concat)
            # Batch Normalization
            if self.use_batch_norm:
                conv_l = BatchNormalization()(conv_l)
            # Activation
            conv_l = Activation('relu')(conv_l)
            # Maxpool
            conv_l = GlobalMaxPool2D()(conv_l)

            # Append the final result
            layer_lst.append(conv_l)

        if len(layer_lst) != 1:
            self.concat_l = concatenate(layer_lst)
        else:
            self.concat_l = layer_lst[0]

        # Dropout
        self.concat_l = Dropout(self.dropout_rate)(self.concat_l)

    def write_hyperparam(self):
        # mode: cnn | nb_epoch: 30 | batch: 200 | opt: adam | lr: 0.007 | k_lst: [3, 4, 5] | nb_filters: 100 |
        # emb_dim: 200 | pos_dim: 10 | sent_len: 150 | dropout: 0.8
        # Write k_lst
        k_lst_str = "["
        for kernel in self.kernel_lst:
            k_lst_str += "{} ".format(kernel)
        k_lst_str = k_lst_str.rstrip()
        k_lst_str += "]"
        log_str_1 = "mode: {} | nb_epoch: {} | batch: {} | opt: {} | lr: {} | hidden_unit_size: {}".format(type(self).__name__,
                                                                                                           self.nb_epoch, self.batch_size,
                                                                                                           self.optimizer,
                                                                                                           self.lr_rate,
                                                                                                           self.hidden_unit_size)
        log_str_2 = "k_lst: {} | nb_filters: {} | emb_dim: {} | pos_dim: {} | sent_len: {} | dropout: {}".format(k_lst_str, self.nb_filters,
                                                                                                                 self.emb_dim, self.pos_dim,
                                                                                                                 self.max_sent_len, self.dropout_rate)
        return log_str_1, log_str_2


class BILSTM(CNN):
    def __init__(self, cfg, vocb, d1_vocb, d2_vocb):
        self.cfg = cfg
        self.vocb = vocb
        self.d1_vocb = d1_vocb
        self.d2_vocb = d2_vocb
        self.nb_epoch = cfg.nb_epoch
        self.batch_size = cfg.batch_size
        self.max_sent_len = cfg.max_sent_len
        self.num_classes = cfg.num_classes
        self.emb_dim = cfg.emb_dim
        self.pos_dim = cfg.pos_dim
        self.rnn_dim = cfg.rnn_dim
        self.dropout_rate = cfg.dropout_rate
        self.optimizer = cfg.optimizer
        self.lr_rate = cfg.lr_rate
        self.non_static = cfg.non_static
        self.use_pretrained = cfg.use_pretrained
        self.unk_limit = cfg.unk_limit
        self.use_self_att = cfg.use_self_att
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

    def write_hyperparam(self):
        # mode: cnn | nb_epoch: 30 | batch: 200 | opt: adam | lr: 0.007 | pretrain: True | k_lst: [3, 4, 5] | nb_filters: 100 |
        # emb_dim: 200 | pos_dim: 10 | sent_len: 150 | dropout: 0.8
        # Write k_lst
        log_str_1 = "mode: {} | nb_epoch: {} | batch: {} | opt: {} | lr: {} | pretrain: {}".format(type(self).__name__, self.nb_epoch,
                                                                                                   self.batch_size,
                                                                                                   self.optimizer, self.lr_rate, self.use_pretrained)
        log_str_2 = "rnn_dim: {} | emb_dim: {} | pos_dim: {} | sent_len: {} | dropout: {}".format(self.rnn_dim, self.emb_dim, self.pos_dim,
                                                                                                  self.max_sent_len, self.dropout_rate)
        return log_str_1, log_str_2

    def add_rnn_layer(self):
        self.rnn_l = Bidirectional(LSTM(self.rnn_dim, dropout=self.dropout_rate,
                                        recurrent_dropout=self.dropout_rate, return_sequences=True))(self.emb_concat)
        self.rnn_l = BatchNormalization()(self.rnn_l)

    def add_self_att_layer(self):
        self.att_l = SeqSelfAttention(attention_activation='sigmoid')(self.rnn_l)
        # self.att_l = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        #                         kernel_regularizer=keras.regularizers.l2(1e-4),
        #                         bias_regularizer=keras.regularizers.l1(1e-4),
        #                         attention_regularizer_weight=1e-4,
        #                         name='Attention')(self.rnn_l)
        self.flat_l = Flatten()(self.att_l)
        self.pred_output = Dense(self.num_classes, activation='softmax')(self.flat_l)

    def add_fc_layer(self):
        self.rnn_l = Flatten()(self.rnn_l)
        self.rnn_l = Activation('relu')(self.rnn_l)
        self.pred_output = Dense(self.num_classes, activation='softmax')(self.rnn_l)


class PCNN(CNN):
    def __init__(self, cfg, vocb, d1_vocb, d2_vocb):
        self.cfg = cfg
        self.vocb = vocb
        self.d1_vocb = d1_vocb
        self.d2_vocb = d2_vocb
        self.nb_epoch = cfg.nb_epoch
        self.batch_size = cfg.batch_size
        self.max_sent_len = cfg.max_sent_len
        self.emb_dim = cfg.emb_dim
        self.pos_dim = cfg.pos_dim
        self.kernel_lst = cfg.kernel_lst
        self.nb_filters = cfg.nb_filters
        self.dropout_rate = cfg.dropout_rate
        self.optimizer = cfg.optimizer
        self.lr_rate = cfg.lr_rate
        self.non_static = cfg.non_static
        self.use_pretrained = cfg.use_pretrained
        self.unk_limit = cfg.unk_limit
        self.num_classes = cfg.num_classes
        self.hidden_unit_size = cfg.hidden_unit_size
        self.use_batch_norm = cfg.use_batch_norm
        self.use_l2_reg = cfg.use_l2_reg
        self.reg_coef_conv = cfg.reg_coef_conv
        self.reg_coef_dense = cfg.reg_coef_dense
        self.build_model()

    def add_input_layer(self):
        # Divide into 3 parts (left, mid, right)
        self.input_sent_left = Input(shape=(self.max_sent_len,), dtype='int32')
        self.input_sent_mid = Input(shape=(self.max_sent_len,), dtype='int32')
        self.input_sent_right = Input(shape=(self.max_sent_len,), dtype='int32')

        self.input_d1_left = Input(shape=(self.max_sent_len,), dtype='int32')
        self.input_d1_mid = Input(shape=(self.max_sent_len,), dtype='int32')
        self.input_d1_right = Input(shape=(self.max_sent_len,), dtype='int32')

        self.input_d2_left = Input(shape=(self.max_sent_len,), dtype='int32')
        self.input_d2_mid = Input(shape=(self.max_sent_len,), dtype='int32')
        self.input_d2_right = Input(shape=(self.max_sent_len,), dtype='int32')

    def add_embedding_layer(self):
        # If static, trainable = False. If non-static, trainable = True
        if self.unk_limit < len(self.vocb):
            input_dim_len = self.unk_limit
        else:
            input_dim_len = len(self.vocb)

        if self.use_pretrained:
            # load word matrix
            # word_matrix = load_word_matrix(self.vocb, self.emb_dim, self.unk_limit)
            word_matrix = load_word_matrix_from_txt(self.vocb, self.emb_dim, self.unk_limit, 'pubmed_and_pmc')
            # If static, trainable = False. If non-static, trainable = True
            self.w_emb_left = Embedding(input_dim=input_dim_len, output_dim=self.emb_dim, input_length=self.max_sent_len,
                                        trainable=self.non_static, weights=[word_matrix])(self.input_sent_left)

            self.w_emb_mid = Embedding(input_dim=input_dim_len, output_dim=self.emb_dim, input_length=self.max_sent_len,
                                       trainable=self.non_static, weights=[word_matrix])(self.input_sent_mid)

            self.w_emb_right = Embedding(input_dim=input_dim_len, output_dim=self.emb_dim, input_length=self.max_sent_len,
                                         trainable=self.non_static, weights=[word_matrix])(self.input_sent_right)
        else:
            # If static, trainable = False. If non-static, trainable = True
            self.w_emb_left = Embedding(input_dim=input_dim_len, output_dim=self.emb_dim,
                                        input_length=self.max_sent_len, trainable=self.non_static)(self.input_sent_left)

            self.w_emb_mid = Embedding(input_dim=input_dim_len, output_dim=self.emb_dim,
                                       input_length=self.max_sent_len, trainable=self.non_static)(self.input_sent_mid)

            self.w_emb_right = Embedding(input_dim=input_dim_len, output_dim=self.emb_dim,
                                         input_length=self.max_sent_len, trainable=self.non_static)(self.input_sent_right)

        # Position Embedding
        # d1
        self.d1_emb_left = Embedding(input_dim=len(self.d1_vocb), output_dim=self.pos_dim,
                                     input_length=self.max_sent_len, trainable=True)(self.input_d1_left)
        self.d1_emb_mid = Embedding(input_dim=len(self.d1_vocb), output_dim=self.pos_dim,
                                    input_length=self.max_sent_len, trainable=True)(self.input_d1_mid)
        self.d1_emb_right = Embedding(input_dim=len(self.d1_vocb), output_dim=self.pos_dim,
                                      input_length=self.max_sent_len, trainable=True)(self.input_d1_right)
        # d2
        self.d2_emb_left = Embedding(input_dim=len(self.d2_vocb), output_dim=self.pos_dim,
                                     input_length=self.max_sent_len, trainable=True)(self.input_d2_left)
        self.d2_emb_mid = Embedding(input_dim=len(self.d2_vocb), output_dim=self.pos_dim,
                                    input_length=self.max_sent_len, trainable=True)(self.input_d2_mid)
        self.d2_emb_right = Embedding(input_dim=len(self.d2_vocb), output_dim=self.pos_dim,
                                      input_length=self.max_sent_len, trainable=True)(self.input_d2_right)
        # Concatenation
        self.emb_concat_left = concatenate([self.w_emb_left, self.d1_emb_left, self.d2_emb_left])
        self.emb_concat_mid = concatenate([self.w_emb_mid, self.d1_emb_mid, self.d2_emb_mid])
        self.emb_concat_right = concatenate([self.w_emb_right, self.d1_emb_right, self.d2_emb_right])

    def add_cnn_layer(self):
        # CNN Parts
        layer_lst = []
        for kernel_size in self.kernel_lst:
            # Sharing the filter weight
            if self.use_l2_reg:
                conv_layer = Conv1D(filters=self.nb_filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l2(self.reg_coef_conv))
            else:
                conv_layer = Conv1D(filters=self.nb_filters, kernel_size=kernel_size, padding='same')
            # left, mid, right convolution
            conv_l_left = conv_layer(self.emb_concat_left)
            conv_l_mid = conv_layer(self.emb_concat_mid)
            conv_l_right = conv_layer(self.emb_concat_right)

            # Batch normalization
            if self.use_batch_norm:
                conv_l_left = BatchNormalization()(conv_l_left)
                conv_l_mid = BatchNormalization()(conv_l_mid)
                conv_l_right = BatchNormalization()(conv_l_right)

            # Activation
            conv_l_left = Activation('relu')(conv_l_left)
            conv_l_mid = Activation('relu')(conv_l_mid)
            conv_l_right = Activation('relu')(conv_l_right)

            # Maxpool
            conv_l_left = GlobalMaxPool1D()(conv_l_left)
            conv_l_mid = GlobalMaxPool1D()(conv_l_mid)
            conv_l_right = GlobalMaxPool1D()(conv_l_right)

            # Concat
            layer_lst.append(concatenate([conv_l_left, conv_l_mid, conv_l_right]))

        if len(layer_lst) != 1:
            self.concat_l = concatenate(layer_lst)
        else:
            self.concat_l = layer_lst[0]

        # Dropout
        self.concat_l = Dropout(self.dropout_rate)(self.concat_l)

    def add_fc_layer(self):
        self.fc_l = self.concat_l
        if self.use_l2_reg:
            self.fc_l = Dense(self.hidden_unit_size, kernel_regularizer=l2(self.reg_coef_dense))(self.fc_l)
        else:
            self.fc_l = Dense(self.hidden_unit_size)(self.fc_l)
        if self.use_batch_norm:
            self.fc_l = BatchNormalization()(self.fc_l)
        self.fc_l = Activation('relu')(self.fc_l)
        if self.use_l2_reg:
            self.pred_output = Dense(self.num_classes, kernel_regularizer=l2(self.reg_coef_dense))(self.fc_l)
        else:
            self.pred_output = Dense(self.num_classes)(self.fc_l)
        self.pred_output = Activation('softmax')(self.pred_output)

    def compile_model(self):
        self.model = Model(inputs=[self.input_sent_left, self.input_sent_mid, self.input_sent_right,
                                   self.input_d1_left, self.input_d1_mid, self.input_d1_right,
                                   self.input_d2_left, self.input_d2_mid, self.input_d2_right], outputs=self.pred_output)
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
                           metrics=['accuracy'])

    def train(self, train_data, dev_data):
        # Unpack data
        (tr_sent_left, tr_d1_left, tr_d2_left), (tr_sent_mid, tr_d1_mid, tr_d2_mid), (tr_sent_right, tr_d1_right, tr_d2_right), tr_y = train_data
        (de_sent_left, de_d1_left, de_d2_left), (de_sent_mid, de_d1_mid, de_d2_mid), (de_sent_right, de_d1_right, de_d2_right), de_y = dev_data

        # Write log
        logging.basicConfig(filename=os.path.join(result_dir, 'result.log'),
                            level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', filemode='a')
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler(sys.stdout))  # For print out the result on console
        logger.info('')
        logger.info("#################################### New Start #####################################")
        # Write log for the hyperparameter
        log_str_1, log_str_2 = self.write_hyperparam()
        logger.info(log_str_1)
        logger.info(log_str_2)
        logger.info('')

        # Training
        self.model.fit(x=[tr_sent_left, tr_sent_mid, tr_sent_right, tr_d1_left, tr_d1_mid, tr_d1_right,
                          tr_d2_left, tr_d2_mid, tr_d2_right], y=tr_y, epochs=self.nb_epoch, batch_size=self.batch_size, verbose=1,
                       validation_data=[[de_sent_left, de_sent_mid, de_sent_right, de_d1_left, de_d1_mid, de_d1_right,
                                         de_d2_left, de_d2_mid, de_d2_right], de_y], callbacks=self.cfg.callback_list)

        # load the best result
        self.model.load_weights(os.path.join(result_dir, 'weights.h5'))

        # Metrics for Train data
        pred_tr = self.model.predict(x=[tr_sent_left, tr_sent_mid, tr_sent_right, tr_d1_left, tr_d1_mid, tr_d1_right,
                                        tr_d2_left, tr_d2_mid, tr_d2_right], batch_size=self.batch_size, verbose=1)
        train_f1 = f1_score(np.argmax(tr_y, 1), np.argmax(pred_tr, 1), [1, 2, 3, 4], average='micro')
        train_p = precision_score(np.argmax(tr_y, 1), np.argmax(pred_tr, 1), [1, 2, 3, 4], average='micro')
        train_r = recall_score(np.argmax(tr_y, 1), np.argmax(pred_tr, 1), [1, 2, 3, 4], average='micro')

        # Metrics for Dev data
        pred_de = self.model.predict(x=[de_sent_left, de_sent_mid, de_sent_right, de_d1_left, de_d1_mid, de_d1_right,
                                        de_d2_left, de_d2_mid, de_d2_right], batch_size=self.batch_size, verbose=1)
        val_f1 = f1_score(np.argmax(de_y, 1), np.argmax(pred_de, 1), [1, 2, 3, 4], average='micro')
        val_p = precision_score(np.argmax(de_y, 1), np.argmax(pred_de, 1), [1, 2, 3, 4], average='micro')
        val_r = recall_score(np.argmax(de_y, 1), np.argmax(pred_de, 1), [1, 2, 3, 4], average='micro')

        # Writing the log
        logger.info('##train##, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(train_p, train_r, train_f1))
        logger.info('##dev##,   prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(val_p, val_r, val_f1))

    def evaluate(self, test_data):
        (te_sent_left, te_d1_left, te_d2_left), (te_sent_mid, te_d1_mid, te_d2_mid), (te_sent_right, te_d1_right, te_d2_right), te_y = test_data
        self.model.load_weights(os.path.join(result_dir, 'weights.h5'))
        pred_te = self.model.predict(x=[te_sent_left, te_sent_mid, te_sent_right, te_d1_left, te_d1_mid, te_d1_right,
                                        te_d2_left, te_d2_mid, te_d2_right], batch_size=self.batch_size, verbose=1)
        te_f1 = f1_score(np.argmax(te_y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
        te_p = precision_score(np.argmax(te_y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
        te_r = recall_score(np.argmax(te_y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
        logger = logging.getLogger()
        logger.info('##test##, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(te_p,
                                                                                te_r,
                                                                                te_f1))

        # Save the best result
        save_best_result(type(self).__name__, te_f1, result_dir)


class MC_PCNN(PCNN):
    def __init__(self, cfg, vocb, d1_vocb, d2_vocb):
        self.cfg = cfg
        self.vocb = vocb
        self.d1_vocb = d1_vocb
        self.d2_vocb = d2_vocb
        self.nb_epoch = cfg.nb_epoch
        self.batch_size = cfg.batch_size
        self.max_sent_len = cfg.max_sent_len
        self.emb_dim = cfg.emb_dim
        self.pos_dim = cfg.pos_dim
        self.kernel_lst = cfg.kernel_lst
        self.nb_filters = cfg.nb_filters
        self.dropout_rate = cfg.dropout_rate
        self.optimizer = cfg.optimizer
        self.lr_rate = cfg.lr_rate
        self.non_static = cfg.non_static
        self.unk_limit = cfg.unk_limit
        self.num_classes = cfg.num_classes
        self.hidden_unit_size = cfg.hidden_unit_size
        self.use_batch_norm = cfg.use_batch_norm
        self.use_l2_reg = cfg.use_l2_reg
        self.reg_coef_conv = cfg.reg_coef_conv
        self.reg_coef_dense = cfg.reg_coef_dense
        self.build_model()

    def add_embedding_layer(self):
        # If static, trainable = False. If non-static, trainable = True
        # load word matrix
        word_matrix_lst = load_word_matrix_all(self.vocb, self.emb_dim, self.unk_limit)
        # word_matrix = load_word_matrix(self.vocb, self.emb_dim, self.unk_limit)
        left_emb_lst = []
        mid_emb_lst = []
        right_emb_lst = []
        # Position Embedding
        # d1
        d1_emb_left = Embedding(input_dim=len(self.d1_vocb), output_dim=self.pos_dim,
                                input_length=self.max_sent_len, trainable=True)(self.input_d1_left)
        d1_emb_mid = Embedding(input_dim=len(self.d1_vocb), output_dim=self.pos_dim,
                               input_length=self.max_sent_len, trainable=True)(self.input_d1_mid)
        d1_emb_right = Embedding(input_dim=len(self.d1_vocb), output_dim=self.pos_dim,
                                 input_length=self.max_sent_len, trainable=True)(self.input_d1_right)
        # d2
        d2_emb_left = Embedding(input_dim=len(self.d2_vocb), output_dim=self.pos_dim,
                                input_length=self.max_sent_len, trainable=True)(self.input_d2_left)
        d2_emb_mid = Embedding(input_dim=len(self.d2_vocb), output_dim=self.pos_dim,
                               input_length=self.max_sent_len, trainable=True)(self.input_d2_mid)
        d2_emb_right = Embedding(input_dim=len(self.d2_vocb), output_dim=self.pos_dim,
                                 input_length=self.max_sent_len, trainable=True)(self.input_d2_right)

        for word_matrix in word_matrix_lst:
            # If static, trainable = False. If non-static, trainable = True
            w_emb_left = Embedding(input_dim=len(word_matrix), output_dim=self.emb_dim, input_length=self.max_sent_len,
                                   trainable=self.non_static, weights=[word_matrix])(self.input_sent_left)

            w_emb_mid = Embedding(input_dim=len(word_matrix), output_dim=self.emb_dim, input_length=self.max_sent_len,
                                  trainable=self.non_static, weights=[word_matrix])(self.input_sent_mid)

            w_emb_right = Embedding(input_dim=len(word_matrix), output_dim=self.emb_dim, input_length=self.max_sent_len,
                                    trainable=self.non_static, weights=[word_matrix])(self.input_sent_right)

            # Concatenation
            emb_concat_left = concatenate([w_emb_left, d1_emb_left, d2_emb_left])
            emb_concat_mid = concatenate([w_emb_mid, d1_emb_mid, d2_emb_mid])
            emb_concat_right = concatenate([w_emb_right, d1_emb_right, d2_emb_right])

            # Reshape and append it
            emb_concat_left = Reshape((self.max_sent_len, self.emb_dim + self.pos_dim * 2, 1))(emb_concat_left)
            emb_concat_mid = Reshape((self.max_sent_len, self.emb_dim + self.pos_dim * 2, 1))(emb_concat_mid)
            emb_concat_right = Reshape((self.max_sent_len, self.emb_dim + self.pos_dim * 2, 1))(emb_concat_right)

            left_emb_lst.append(emb_concat_left)
            mid_emb_lst.append(emb_concat_mid)
            right_emb_lst.append(emb_concat_right)

        # concat all the five embedding
        self.emb_concat_left = concatenate(left_emb_lst)
        self.emb_concat_mid = concatenate(mid_emb_lst)
        self.emb_concat_right = concatenate(right_emb_lst)

    def add_cnn_layer(self):
        # CNN Parts
        layer_lst = []
        for kernel_size in self.kernel_lst:
            # Sharing the filter weight
            if self.use_l2_reg:
                conv_layer = Conv2D(self.nb_filters, kernel_size=(kernel_size, self.emb_dim + self.pos_dim * 2), padding='same',
                                    strides=(1, self.emb_dim + self.pos_dim * 2), kernel_regularizer=l2(self.reg_coef_conv))
            else:
                conv_layer = Conv2D(self.nb_filters, kernel_size=(kernel_size, self.emb_dim + self.pos_dim * 2), padding='same',
                                    strides=(1, self.emb_dim + self.pos_dim * 2))

            # zero padding on embedding layer, only on height
            # if kernel_size % 2 == 0:
            #     padding_size = int(kernel_size / 2)
            # else:
            #     padding_size = int((kernel_size - 1) / 2)
            # padded_emb_concat_left = ZeroPadding2D((padding_size, 0))(self.emb_concat_left)
            # padded_emb_concat_mid = ZeroPadding2D((padding_size, 0))(self.emb_concat_mid)
            # padded_emb_concat_right = ZeroPadding2D((padding_size, 0))(self.emb_concat_right)

            # left, mid, right convolution
            conv_l_left = conv_layer(self.emb_concat_left)
            conv_l_mid = conv_layer(self.emb_concat_mid)
            conv_l_right = conv_layer(self.emb_concat_right)

            # Batch normalization
            if self.use_batch_norm:
                conv_l_left = BatchNormalization()(conv_l_left)
                conv_l_mid = BatchNormalization()(conv_l_mid)
                conv_l_right = BatchNormalization()(conv_l_right)

            # Activation
            conv_l_left = Activation('relu')(conv_l_left)
            conv_l_mid = Activation('relu')(conv_l_mid)
            conv_l_right = Activation('relu')(conv_l_right)

            # Maxpool
            conv_l_left = GlobalMaxPool2D()(conv_l_left)
            conv_l_mid = GlobalMaxPool2D()(conv_l_mid)
            conv_l_right = GlobalMaxPool2D()(conv_l_right)

            # Concat
            layer_lst.append(concatenate([conv_l_left, conv_l_mid, conv_l_right]))

        if len(layer_lst) != 1:
            self.concat_l = concatenate(layer_lst)
        else:
            self.concat_l = layer_lst[0]

        # Dropout at the last layer
        self.concat_l = Dropout(self.dropout_rate)(self.concat_l)

    def add_fc_layer(self):
        self.fc_l = self.concat_l
        if self.use_l2_reg:
            self.fc_l = Dense(self.hidden_unit_size, kernel_regularizer=l2(self.reg_coef_dense))(self.fc_l)
        else:
            self.fc_l = Dense(self.hidden_unit_size)(self.fc_l)
        if self.use_batch_norm:
            self.fc_l = BatchNormalization()(self.fc_l)
        self.fc_l = Activation('relu')(self.fc_l)
        self.fc_l = Dropout(self.dropout_rate)(self.fc_l)  # Put Dropout between fc_layer
        if self.use_l2_reg:
            self.pred_output = Dense(self.num_classes, kernel_regularizer=l2(self.reg_coef_dense))(self.fc_l)
        else:
            self.pred_output = Dense(self.num_classes)(self.fc_l)
        self.pred_output = Activation('softmax')(self.pred_output)

    def write_hyperparam(self):
        # mode: cnn | nb_epoch: 30 | batch: 200 | opt: adam | lr: 0.007 | pretrain: True | k_lst: [3, 4, 5] | nb_filters: 100 |
        # emb_dim: 200 | pos_dim: 10 | sent_len: 150 | dropout: 0.8
        # Write k_lst
        k_lst_str = "["
        for kernel in self.kernel_lst:
            k_lst_str += "{} ".format(kernel)
        k_lst_str = k_lst_str.rstrip()
        k_lst_str += "]"
        log_str_1 = "mode: {} | nb_epoch: {} | batch: {} | opt: {} | lr: {} | hidden_unit_size: {}".format(type(self).__name__,
                                                                                                           self.nb_epoch, self.batch_size,
                                                                                                           self.optimizer,
                                                                                                           self.lr_rate,
                                                                                                           self.hidden_unit_size)
        log_str_2 = "k_lst: {} | nb_filters: {} | emb_dim: {} | pos_dim: {} | sent_len: {} | dropout: {}".format(k_lst_str, self.nb_filters,
                                                                                                                 self.emb_dim, self.pos_dim,
                                                                                                                 self.max_sent_len, self.dropout_rate)
        return log_str_1, log_str_2


class MC_PCNN_ATT(MC_PCNN):
    def add_cnn_layer(self):
        # CNN Parts
        layer_lst = []
        for kernel_size in self.kernel_lst:
            # Sharing the filter weight
            if self.use_l2_reg:
                conv_layer = Conv2D(self.nb_filters, kernel_size=(kernel_size, self.emb_dim + self.pos_dim * 2), padding='same',
                                    strides=(1, self.emb_dim + self.pos_dim * 2),
                                    kernel_regularizer=l2(self.reg_coef_conv))
            else:
                conv_layer = Conv2D(self.nb_filters, kernel_size=(kernel_size, self.emb_dim + self.pos_dim * 2), padding='same',
                                    strides=(1, self.emb_dim + self.pos_dim * 2))

            # zero padding on embedding layer, only on height
            # if kernel_size % 2 == 0:
            #     padding_size = int(kernel_size / 2)
            # else:
            #     padding_size = int((kernel_size - 1) / 2)
            # padded_emb_concat_left = ZeroPadding2D((padding_size, 0))(self.emb_concat_left)
            # padded_emb_concat_mid = ZeroPadding2D((padding_size, 0))(self.emb_concat_mid)
            # padded_emb_concat_right = ZeroPadding2D((padding_size, 0))(self.emb_concat_right)

            # left, mid, right convolution
            conv_l_left = conv_layer(self.emb_concat_left)
            conv_l_mid = conv_layer(self.emb_concat_mid)
            conv_l_right = conv_layer(self.emb_concat_right)

            # Batch normalization
            if self.use_batch_norm:
                conv_l_left = BatchNormalization()(conv_l_left)
                conv_l_mid = BatchNormalization()(conv_l_mid)
                conv_l_right = BatchNormalization()(conv_l_right)

            # Activation
            conv_l_left = Activation('relu')(conv_l_left)
            conv_l_mid = Activation('relu')(conv_l_mid)
            conv_l_right = Activation('relu')(conv_l_right)

            # Maxpool
            conv_l_left = GlobalMaxPool2D()(conv_l_left)
            conv_l_mid = GlobalMaxPool2D()(conv_l_mid)
            conv_l_right = GlobalMaxPool2D()(conv_l_right)

            # Concat
            conv_l_left = Reshape((1, self.nb_filters))(conv_l_left)
            conv_l_mid = Reshape((1, self.nb_filters))(conv_l_mid)
            conv_l_right = Reshape((1, self.nb_filters))(conv_l_right)
            conv_concat = concatenate([conv_l_left, conv_l_mid, conv_l_right], axis=-2)

            # Add dropout until Self-Attention
            conv_concat = Dropout(self.dropout_rate)(conv_concat)

            # TODO: 각 윈도우별로 self attention을 붙여야 하는지, 모든 window를 concat한 [None, 3, 400]의 상태에서 해야할지
            conv_concat = SeqSelfAttention(units=32, attention_activation='sigmoid')(conv_concat)  # Base self attention
            # conv_concat = SeqSelfAttention(units=32,
            #                                attention_activation='sigmoid',
            #                                kernel_regularizer=l2(1e-6),
            #                                bias_regularizer=l1(1e-6),
            #                                attention_regularizer_weight=1e-6)(conv_concat)
            conv_concat = Flatten()(conv_concat)
            layer_lst.append(conv_concat)

        if len(layer_lst) != 1:
            self.concat_l = concatenate(layer_lst)
        else:
            self.concat_l = layer_lst[0]

        # Dropout at the last layer
        self.concat_l = Dropout(self.dropout_rate)(self.concat_l)
