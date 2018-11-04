from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import os

result_dir = 'result'

# 1. Training settings
train_mode = 'cnn'  # [cnn, pcnn, mccnn, mcpcnn, rnn]
nb_epoch = 100
batch_size = 128
learning_rate = 0.0005
optimizer = 'adam'
use_pretrained = True  # If you're using pretrained, emb_dim will be 200 for PubMed-and-PMC-w2v.bin (http://evexdb.org/pmresources/vec-space-models/)
dev_size = 0.1
hidden_unit_size = 256
use_batch_norm = False
dropout_rate = 0.5

# l2 regularizer setting
use_l2_reg = False
reg_coef_conv = 0.0001
reg_coef_dense = 0.0001

# 2. CNN specific
kernel_lst = [3, 5, 7, 9]  # [3, 5, 7]
nb_filters = 200

# 3. RNN specific
rnn_dim = 200  # Dimension for output of LSTM

# 4. Model common settings
emb_dim = 200
pos_dim = 10
max_sent_len = 150
num_classes = 5
unk_limit = 5000

# 5. Self attention
use_self_att = False

# callback function
# CallBack setting
callback_list = [
    # 1. Early Stopping Callback
    EarlyStopping(monitor='val_loss', patience=6),
    # 2. Model Checkpoint
    ModelCheckpoint(filepath=os.path.join(result_dir, 'weights.h5'), monitor='val_loss', save_best_only=True),
    # 3. Reducing Learning rate automatically
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5),  # Reduce the lr_rate into 10%
    # 4. Tensorboard callback
    # TensorBoard(log_dir=tf_board_dir, histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
]