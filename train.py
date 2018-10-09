from load_data_ddi import load_data
from model import CNN, MCCNN, BILSTM

########### Hyperparameter ###########
nb_epoch = 10
batch_size = 32
max_sent_len = 50
unk_limit = 4000
learning_rate = 0.001
kernel_lst = [3, 4, 5]
emb_dim = 50
pos_dim = 10
rnn_dim = 100
nb_filters = 16
dropout_rate = 0.1
######################################

if __name__ == '__main__':
    (sentences2idx, entity_pos_lst, y), (vocb, vocb_inv), (sentences, drug1_lst,
                                                           drug2_lst, rel_lst) = load_data(unk_limit=unk_limit, max_sent_len=max_sent_len)
    # model = MCCNN(max_sent_len=max_sent_len,
    #             vocb=vocb,
    #             emb_dim=emb_dim,
    #             pos_dim=pos_dim,
    #             kernel_lst=kernel_lst,
    #             nb_filters=nb_filters,
    #             dropout_rate=dropout_rate,
    #             optimizer='adam',
    #             lr_rate=learning_rate,
    #             use_pretrained=False)
    # model = CNN(max_sent_len=max_sent_len,
    #             vocb=vocb,
    #             emb_dim=emb_dim,
    #             pos_dim=pos_dim,
    #             kernel_lst=kernel_lst,
    #             nb_filters=nb_filters,
    #             dropout_rate=dropout_rate,
    #             optimizer='adam',
    #             non_static=True,
    #             lr_rate=learning_rate,
    #             use_pretrained=False)
    model = BILSTM(max_sent_len=max_sent_len,
                   vocb=vocb,
                   emb_dim=emb_dim,
                   pos_dim=pos_dim,
                   rnn_dim=rnn_dim,
                   dropout_rate=0.2,
                   optimizer='adam',
                   non_static=True,
                   lr_rate=0.001,
                   use_pretrained=False)
    model.show_model_summary()
    model.save_model()
    model.train(sentence=sentences2idx, pos_lst=entity_pos_lst, y=y, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.2)
