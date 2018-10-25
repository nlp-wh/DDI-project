import numpy as np
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, Embedding, Input, concatenate, add, Bidirectional, LSTM, BatchNormalization, \
    GlobalMaxPool1D
from keras import Model
from keras.optimizers import Adam, Adadelta, RMSprop, Adagrad
from sklearn.metrics import f1_score, recall_score, precision_score

import os
import sys
import logging

from load_data_ddi import load_word_matrix, load_test_pair_id
from seq_self_attention import SeqSelfAttention

# Make the directory for saving model, weight, log
result_dir = 'result'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


class PCNN(object):
    def __init__(self,
                 max_sent_len,
                 vocb,
                 d1_vocb,
                 d2_vocb,
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
        self.d1_vocb = d1_vocb
        self.d2_vocb = d2_vocb
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
        if self.use_pretrained:
            # load word matrix
            word_matrix = load_word_matrix(self.vocb, self.emb_dim, self.unk_limit)
            # If static, trainable = False. If non-static, trainable = True
            self.w_emb_left = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim, input_length=self.max_sent_len,
                                        trainable=self.non_static, weights=[word_matrix])(self.input_sent_left)

            self.w_emb_mid = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim, input_length=self.max_sent_len,
                                       trainable=self.non_static, weights=[word_matrix])(self.input_sent_mid)

            self.w_emb_right = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim, input_length=self.max_sent_len,
                                         trainable=self.non_static, weights=[word_matrix])(self.input_sent_right)
        else:
            # If static, trainable = False. If non-static, trainable = True
            self.w_emb_left = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim,
                                        input_length=self.max_sent_len, trainable=self.non_static)(self.input_sent_left)

            self.w_emb_mid = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim,
                                       input_length=self.max_sent_len, trainable=self.non_static)(self.input_sent_mid)

            self.w_emb_right = Embedding(input_dim=len(self.vocb), output_dim=self.emb_dim,
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
            # left, mid, right convolution
            conv_l_left = Conv1D(filters=self.nb_filters, kernel_size=kernel_size, padding='same', activation='relu')(self.emb_concat_left)
            conv_l_mid = Conv1D(filters=self.nb_filters, kernel_size=kernel_size, padding='same', activation='relu')(self.emb_concat_mid)
            conv_l_right = Conv1D(filters=self.nb_filters, kernel_size=kernel_size, padding='same', activation='relu')(self.emb_concat_right)

            # Maxpool
            pool_l_left = GlobalMaxPool1D()(conv_l_left)
            pool_l_mid = GlobalMaxPool1D()(conv_l_mid)
            pool_l_right = GlobalMaxPool1D()(conv_l_right)

            # Dropout
            drop_l_left = Dropout(self.dropout_rate)(pool_l_left)
            drop_l_mid = Dropout(self.dropout_rate)(pool_l_mid)
            drop_l_right = Dropout(self.dropout_rate)(pool_l_right)

            # Concat
            layer_lst.append(concatenate([drop_l_left, drop_l_mid, drop_l_right]))

        self.concat_l = concatenate(layer_lst)

    def add_fc_layer(self):
        '''
        self.fc_l_1 = Dense(300, activation='relu')(self.concat_l)
        self.fc_l_1 = Dropout(self.dropout_rate)(self.fc_l_1)
        self.pred_output = Dense(self.num_classes, activation='softmax')(self.fc_l_1)
        '''
        self.pred_output = Dense(self.num_classes, activation='softmax')(self.concat_l)

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

    def save_model(self):
        # Save the model into the result directory
        model_json = self.model.to_json()
        with open(os.path.join(result_dir, 'model.json'), "w", encoding='utf-8') as json_file:
            json_file.write(model_json)
        print('Save model.json')

    def train(self, nb_epoch, batch_size, train_data, dev_data):
        # Unpack data
        (tr_sent_left, tr_d1_left, tr_d2_left), (tr_sent_mid, tr_d1_mid, tr_d2_mid), (tr_sent_right, tr_d1_right, tr_d2_right), tr_y = train_data
        (de_sent_left, de_d1_left, de_d2_left), (de_sent_mid, de_d1_mid, de_d2_mid), (de_sent_right, de_d1_right, de_d2_right), de_y = dev_data
        # (tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_y) = train_data
        # (de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_y) = dev_data

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
            train_history = self.model.fit(x=[tr_sent_left, tr_sent_mid, tr_sent_right, tr_d1_left, tr_d1_mid, tr_d1_right,
                                              tr_d2_left, tr_d2_mid, tr_d2_right], y=tr_y, epochs=1, batch_size=batch_size, verbose=1)
            # Metrics for Train data
            pred_tr = self.model.predict(x=[tr_sent_left, tr_sent_mid, tr_sent_right, tr_d1_left, tr_d1_mid, tr_d1_right,
                                            tr_d2_left, tr_d2_mid, tr_d2_right], batch_size=batch_size, verbose=1)
            train_loss = train_history.history['loss'][0]
            train_acc = train_history.history['acc'][0]
            train_f1 = f1_score(np.argmax(tr_y, 1), np.argmax(pred_tr, 1), [1, 2, 3, 4], average='micro')
            train_p = precision_score(np.argmax(tr_y, 1), np.argmax(pred_tr, 1), [1, 2, 3, 4], average='micro')
            train_r = recall_score(np.argmax(tr_y, 1), np.argmax(pred_tr, 1), [1, 2, 3, 4], average='micro')

            # Metrics for Dev data
            pred_de = self.model.predict(x=[de_sent_left, de_sent_mid, de_sent_right, de_d1_left, de_d1_mid, de_d1_right,
                                            de_d2_left, de_d2_mid, de_d2_right], batch_size=batch_size, verbose=1)
            val_f1 = f1_score(np.argmax(de_y, 1), np.argmax(pred_de, 1), [1, 2, 3, 4], average='micro')
            val_p = precision_score(np.argmax(de_y, 1), np.argmax(pred_de, 1), [1, 2, 3, 4], average='micro')
            val_r = recall_score(np.argmax(de_y, 1), np.argmax(pred_de, 1), [1, 2, 3, 4], average='micro')
            # Writing the log
            logger.info('##train##, epoch: {:2d}, loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format((i + 1),
                                                                                                                              train_loss,
                                                                                                                              train_acc,
                                                                                                                              train_p,
                                                                                                                              train_r,
                                                                                                                              train_f1))
            logger.info('##dev##,   epoch: {:2d}, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format((i + 1),
                                                                                                   val_p,
                                                                                                   val_r,
                                                                                                   val_f1))
            # Saving the weight if it is better than before (early-stopping)
            if max_val_f1 < val_f1:
                max_val_f1 = val_f1
                logging.info("[{}th epoch, Better performance! Update the weight!]".format(i + 1))
                self.model.save_weights(os.path.join(result_dir, 'weights.h5'))

    def evaluate(self, test_data, batch_size):
        (te_sent_left, te_d1_left, te_d2_left), (te_sent_mid, te_d1_mid, te_d2_mid), (te_sent_right, te_d1_right, te_d2_right), te_y = test_data
        self.model.load_weights(os.path.join(result_dir, 'weights.h5'))
        pred_te = self.model.predict(x=[te_sent_left, te_sent_mid, te_sent_right, te_d1_left, te_d1_mid, te_d1_right,
                                        te_d2_left, te_d2_mid, te_d2_right], batch_size=batch_size, verbose=1)
        te_f1 = f1_score(np.argmax(te_y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
        te_p = precision_score(np.argmax(te_y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
        te_r = recall_score(np.argmax(te_y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
        logger = logging.getLogger()
        logger.info('##test##, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(te_p,
                                                                                te_r,
                                                                                te_f1))

    def show_model_summary(self):
        print(self.model.summary(line_length=100))

    def predict(self, sentences2idx, d1_lst, d2_lst, batch_size, one_hot=True):
        self.model.load_weights(os.path.join(result_dir, 'weights.h5'))
        y_pred = self.model.predict(x=[sentences2idx, d1_lst, d2_lst], batch_size=batch_size)
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

    @staticmethod
    def make_output_file(y_pred):
        pair_id_lst = load_test_pair_id()
        with open('output.tsv', 'w', encoding='utf-8') as f:
            assert len(y_pred) == len(pair_id_lst)
            for pair_id, y in zip(pair_id_lst, y_pred):
                line = "{}\t{}\t{}\n".format("DDI2013", pair_id, y)
                f.write(line)