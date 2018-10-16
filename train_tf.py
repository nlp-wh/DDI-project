from load_data_ddi import load_data
from model_tf import CNN, batch_loader
import tensorflow as tf
import sys

########### Hyperparameter ###########
# 1. Training settings
train_mode = 'cnn'
nb_epoch = 10
batch_size = 200
learning_rate = 0.001
optimizer = 'adam'
use_pretrained = False

# 2. CNN specific
kernel_lst = [3, 4, 5]  # [3, 4, 5]
nb_filters = 200

# 3. RNN specific
rnn_dim = 300  # Dimension for output of LSTM

# 4. Model common settings
emb_dim = 300
pos_dim = 10
max_sent_len = 150
num_classes = 5
unk_limit = 8000
dropout_keep_prob = 0.5

# 5. Self attention
use_self_att = False
######################################


if __name__ == '__main__':
    (tr_sentences2idx, tr_entity_pos_lst, tr_y), (te_sentences2idx, te_entity_pos_lst, te_y), (vocb, vocb_inv), \
    (tr_sentences, tr_drug1_lst, tr_drug2_lst, tr_rel_lst), \
    (te_sentences, te_drug1_lst, te_drug2_lst, te_rel_lst) = load_data(unk_limit=unk_limit,
                                                                       max_sent_len=max_sent_len)
    model = CNN(max_sent_len=max_sent_len,
                vocb=vocb,
                emb_dim=emb_dim,
                pos_dim=pos_dim,
                kernel_lst=kernel_lst,
                nb_filters=nb_filters,
                optimizer=optimizer,
                non_static=True,
                lr_rate=learning_rate,
                use_pretrained=use_pretrained,
                unk_limit=unk_limit,
                num_classes=num_classes)

    # Initialize
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training
    for i in range(nb_epoch):
        train_data = list(zip(tr_sentences2idx, tr_entity_pos_lst, tr_y))
        train_batches = batch_loader(train_data, batch_size, shuffle=True)
        for batch in train_batches:
            batch_x, batch_pos, batch_y = zip(*batch)
            step, loss, acc = model.train(sess=sess, sentence=batch_x, pos_lst=batch_pos, y=batch_y,
                                          dropout_keep_prob=dropout_keep_prob)
            print("epoch: {}, loss: {:.4f}, acc: {:.4f}".format(i + 1, loss, acc))

    # Evaluation
    # model.evaluate(sess=sess, sentence=te_sentences2idx, pos_lst=te_entity_pos_lst, y=te_y)
