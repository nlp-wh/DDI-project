from load_data_ddi import load_data, sentence_split_for_pcnn
from pcnn import PCNN

########### Hyperparameter ###########
# 1. Training settings
train_mode = 'cnn'
nb_epoch = 10
batch_size = 50
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

    (tr_sent_left, tr_d1_left, tr_d2_left), (tr_sent_mid, tr_d1_mid, tr_d2_mid), (tr_sent_right, tr_d1_right, tr_d2_right) = \
        sentence_split_for_pcnn(sentences2idx=tr_sentences2idx, d1_pos_lst=tr_d1_pos_lst, d2_pos_lst=tr_d1_pos_lst,
                                pos_tuple_lst=tr_pos_tuple_lst, max_sent_len=max_sent_len)

    (de_sent_left, de_d1_left, de_d2_left), (de_sent_mid, de_d1_mid, de_d2_mid), (de_sent_right, de_d1_right, de_d2_right) = \
        sentence_split_for_pcnn(sentences2idx=de_sentences2idx, d1_pos_lst=de_d1_pos_lst, d2_pos_lst=de_d1_pos_lst,
                                pos_tuple_lst=de_pos_tuple_lst, max_sent_len=max_sent_len)

    (te_sent_left, te_d1_left, te_d2_left), (te_sent_mid, te_d1_mid, te_d2_mid), (te_sent_right, te_d1_right, te_d2_right) = \
        sentence_split_for_pcnn(sentences2idx=te_sentences2idx, d1_pos_lst=te_d1_pos_lst, d2_pos_lst=te_d1_pos_lst,
                                pos_tuple_lst=te_pos_tuple_lst, max_sent_len=max_sent_len)

    model = PCNN(max_sent_len=max_sent_len,
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

    model.show_model_summary()
    model.save_model()
    model.train(nb_epoch=nb_epoch, batch_size=batch_size, train_data=(
        (tr_sent_left, tr_d1_left, tr_d2_left), (tr_sent_mid, tr_d1_mid, tr_d2_mid), (tr_sent_right, tr_d1_right, tr_d2_right), tr_y),
                dev_data=(
                    (de_sent_left, de_d1_left, de_d2_left), (de_sent_mid, de_d1_mid, de_d2_mid), (de_sent_right, de_d1_right, de_d2_right), de_y))
    model.evaluate(
        test_data=((te_sent_left, te_d1_left, te_d2_left), (te_sent_mid, te_d1_mid, te_d2_mid), (te_sent_right, te_d1_right, te_d2_right), te_y),
        batch_size=batch_size)
