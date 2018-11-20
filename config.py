from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import os
import random

result_dir = 'result'

# 1. Training settings
train_mode = 'mcpcnn_att'  # [cnn, pcnn, mccnn, mcpcnn, rnn, mcpcnn_att]
nb_epoch = 100
batch_size = 64
learning_rate = 0.0005
optimizer = 'adam'
use_pretrained = True
dev_size = 0.1
hidden_unit_size = 256
use_batch_norm = True
dropout_rate = 0.5

# l2 regularizer setting
use_l2_reg = True
reg_coef_conv = 1e-7
reg_coef_dense = 1e-7

# 2. CNN specific
kernel_lst = [3, 5, 7, 9]  # [3, 5, 7]
nb_filters = random.choice([80, 100, 120])

# 3. RNN specific
rnn_dim = 200  # Dimension for output of LSTM

# 4. Model common settings
emb_dim = 200
pos_dim = random.choice([10, 20])
max_sent_len = 150
num_classes = 5
unk_limit = 3000

# 5. Self attention
use_self_att = False

# callback function
# CallBack setting
callback_list = [
    # 1. Early Stopping Callback
    EarlyStopping(monitor='val_loss', patience=4),
    # 2. Model Checkpoint
    ModelCheckpoint(filepath=os.path.join(result_dir, 'weights.h5'), monitor='val_loss', save_best_only=True),
    # 3. Reducing Learning rate automatically
    ReduceLROnPlateau(monitor='val_acc', patience=1, factor=0.2),  # Reduce the lr_rate into 10%
    # 4. Tensorboard callback
    # TensorBoard(log_dir=tf_board_dir, histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
]
