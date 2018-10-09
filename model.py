import numpy as np
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, Activation, Embedding, Input, concatenate, add
from keras import Model, regularizers
from keras.optimizers import Adam, Adadelta, RMSprop, Adagrad

import os
import sys
import logging

# Make the directory for saving model, weight, log
result_dir = 'result'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


def load_word_matrix(vocb, emb_dim):
    embedding_index = dict()
    with open('glove.6B.{}d.txt'.format(emb_dim), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    word_matrix = np.zeros((len(vocb), emb_dim))
    cnt = 0
    for word, i in vocb.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            word_matrix[i] = embedding_vector
        else:
            word_matrix[i] = np.random.uniform(-1.0, 1.0, emb_dim)
            cnt += 1
    print('{} words not in glove'.format(cnt))
    return word_matrix


class CNN(object):
    def __init__(self,
                 max_sent_len,
                 vocb,
                 emb_dim=100,
                 pos_dim=10,
                 kernel_lst=[3, 4, 5],
                 nb_filters=100,
                 dropout_rate=0.2,
                 optimizer='adam',
                 lr_rate=0.001,
                 non_static=True,
                 use_pretrained=False):
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
            word_matrix = load_word_matrix(self.vocb, self.emb_dim)
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
        self.pred_output = Dense(4, activation='softmax')(self.flat_l)

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
                           metrics=['accuracy'])

    def save_model(self):
        # Save the model into the result directory
        model_json = self.model.to_json()
        with open(os.path.join(result_dir, 'model.json'), "w", encoding='utf-8') as json_file:
            json_file.write(model_json)
        print('Save model.json')

    def train(self, sentence, pos_lst, y, nb_epoch, batch_size, validation_split, save_weights=False):
        # Write log
        logging.basicConfig(filename=os.path.join(result_dir, 'cnn.log'),
                            level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', filemode='a')
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler(sys.stdout))  # For print out the result on console
        logger.info('')
        logger.info("#################################### New Start #####################################")

        max_val_acc = 0
        for i in range(nb_epoch):
            # Training
            train_history = self.model.fit(x=[sentence, pos_lst], y=y, epochs=1, batch_size=batch_size, verbose=1, validation_split=validation_split)
            # Writing the log
            train_loss = train_history.history['loss'][0]
            train_acc = train_history.history['acc'][0]
            val_loss = train_history.history['val_loss'][0]
            val_acc = train_history.history['val_acc'][0]
            logger.info(
                '##train##, epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format((i + 1),
                                                                                                                        train_loss,
                                                                                                                        train_acc,
                                                                                                                        val_loss,
                                                                                                                        val_acc))
            # Saving the weight if it is better than before (early-stopping)
            if max_val_acc < val_acc:
                max_val_acc = val_acc
                logging.info("[{}th epoch, Better performance! Update the weight!]".format(i + 1))
                self.model.save_weights(os.path.join(result_dir, 'weights.h5'))

    def evaluate(self, x_test, y_test, batch_size):
        self.model.load_weights(os.path.join(result_dir, 'weights.h5'))
        pred_test = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
        logger = logging.getLogger()
        logger.info(
            '##test##,  test_loss: {:.4f}, test_acc: {:.4f}'.format(pred_test[0], pred_test[1]))

    def show_model_summary(self):
        print(self.model.summary())


class MCCNN(CNN):
    def __init__(self,
                 max_sent_len,
                 vocb,
                 emb_dim=100,
                 pos_dim=10,
                 kernel_lst=[3, 4, 5],
                 nb_filters=100,
                 dropout_rate=0.2,
                 optimizer='adam',
                 lr_rate=0.001,
                 use_pretrained=False):
        self.max_sent_len = max_sent_len
        self.vocb = vocb
        self.emb_dim = emb_dim
        self.pos_dim = pos_dim
        self.kernel_lst = kernel_lst
        self.nb_filters = nb_filters
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.lr_rate = lr_rate
        self.use_pretrained = use_pretrained
        self.build_model()

    def add_embedding_layer(self):
        # If static, trainable = False. If non-static, trainable = True
        if self.use_pretrained:
            # load word matrix
            word_matrix = load_word_matrix(self.vocb, self.emb_dim)
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
