from load_data_ddi import load_data
from model import CNN, MCCNN, BILSTM

########### Hyperparameter ###########
# 1. Training settings
train_mode = 'cnn'
nb_epoch = 10
batch_size = 200
learning_rate = 0.001
optimizer = 'adam'
use_pretrained = False  # If you're using pretrained, emb_dim will be 200 for PubMed-and-PMC-w2v.bin (http://evexdb.org/pmresources/vec-space-models/)
dev_size = 0.1

# 2. CNN specific
kernel_lst = [3, 5, 7]
nb_filters = 100

# 3. RNN specific
rnn_dim = 200  # Dimension for output of LSTM

# 4. Model common settings
emb_dim = 200
pos_dim = 20
max_sent_len = 150
num_classes = 5
unk_limit = 8000
dropout_rate = 0.5

# 5. Self attention
use_self_att = False
######################################


if __name__ == '__main__':
    (tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_pos_tuple_lst, tr_y), \
        (de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_pos_tuple_lst, de_y), \
        (te_sentences2idx, te_d1_pos_lst, te_d2_pos_lst, te_pos_tuple_lst, te_y), \
        (vocb, vocb_inv), (d1_vocb, d2_vocb) = load_data(unk_limit=unk_limit, max_sent_len=max_sent_len, dev_size=dev_size)

    if train_mode.lower() == 'cnn':
        model = CNN(max_sent_len=max_sent_len,
                    vocb=vocb,
                    d1_vocb=d1_vocb,
                    d2_vocb=d2_vocb,
                    emb_dim=emb_dim,
                    pos_dim=pos_dim,
                    kernel_lst=kernel_lst,
                    nb_filters=nb_filters,
                    dropout_rate=dropout_rate,
                    optimizer=optimizer,
                    non_static=True,
                    lr_rate=learning_rate,
                    use_pretrained=use_pretrained,
                    unk_limit=unk_limit,
                    num_classes=num_classes)

    elif train_mode.lower() == 'mccnn':
        model = MCCNN(max_sent_len=max_sent_len,
                      vocb=vocb,
                      d1_vocb=d1_vocb,
                      d2_vocb=d2_vocb,
                      emb_dim=emb_dim,
                      pos_dim=pos_dim,
                      kernel_lst=kernel_lst,
                      nb_filters=nb_filters,
                      dropout_rate=dropout_rate,
                      optimizer=optimizer,
                      lr_rate=learning_rate,
                      use_pretrained=use_pretrained,
                      unk_limit=unk_limit,
                      num_classes=num_classes)

    elif train_mode.lower() == 'rnn':
        model = BILSTM(max_sent_len=max_sent_len,
                       vocb=vocb,
                       d1_vocb=d1_vocb,
                       d2_vocb=d2_vocb,
                       emb_dim=emb_dim,
                       pos_dim=pos_dim,
                       rnn_dim=rnn_dim,
                       dropout_rate=dropout_rate,
                       optimizer=optimizer,
                       non_static=True,
                       lr_rate=learning_rate,
                       use_pretrained=use_pretrained,
                       unk_limit=unk_limit,
                       num_classes=num_classes,
                       use_self_att=use_self_att)

    else:
        raise Exception("Wrong Training Model")
    model.show_model_summary()
    model.save_model()
    model.train(nb_epoch=nb_epoch, batch_size=batch_size, train_data=(tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_y),
                dev_data=(de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_y))
    model.evaluate(sentences2idx=te_sentences2idx, d1_lst=te_d1_pos_lst, d2_lst=te_d2_pos_lst, y=te_y, batch_size=batch_size)
