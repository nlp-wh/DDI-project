from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import os
import random

result_dir = 'result'


class Config(object):
    def __init__(self):
        # 1. Training settings
        self.train_mode = 'mcpcnn_att'  # [cnn, pcnn, mccnn, mcpcnn, rnn, mcpcnn_att]
        self.nb_epoch = 100
        self.batch_size = 64
        self.lr_rate = 0.0005
        self.optimizer = 'adam'
        self.non_static = True
        self.use_pretrained = True
        self.dev_size = 0.1
        self.hidden_unit_size = 128
        self.use_batch_norm = True
        self.dropout_rate = random.choice([0.4, 0.45, 0.5])

        # l2 regularizer setting
        self.use_l2_reg = True
        self.reg_coef_conv = 1e-7
        self.reg_coef_dense = 1e-7

        # 2. CNN specific
        self.kernel_lst = [3, 5, 7, 9]  # [3, 5, 7]
        self.nb_filters = random.choice([100, 120])

        # 3. RNN specific
        self.rnn_dim = 200  # Dimension for output of LSTM

        # 4. Model common settings
        self.emb_dim = 200
        self.pos_dim = random.choice([5, 10, 20])
        self.max_sent_len = 150
        self.num_classes = 5
        self.unk_limit = 3000

        # 5. Self attention
        self.use_self_att = False

        # CallBack setting
        self.callback_list = [
            # 1. Early Stopping Callback
            EarlyStopping(monitor='val_loss', patience=4),
            # 2. Model Checkpoint
            ModelCheckpoint(filepath=os.path.join(result_dir, 'weights.h5'), monitor='val_loss', save_best_only=True),
            # 3. Reducing Learning rate automatically
            ReduceLROnPlateau(monitor='val_acc', patience=1, factor=0.2),  # Reduce the lr_rate into 10%
            # 4. Tensorboard callback
            # TensorBoard(log_dir=tf_board_dir, histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
        ]
