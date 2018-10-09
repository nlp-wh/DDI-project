from load_data_ade import load_data
from model import CNN, MCCNN

########### Hyperparameter ###########
nb_epoch = 20
batch_size = 32
max_sent_len = 50
unk_limit = 10000
learning_rate = 0.001
kernel_lst = [3, 4, 5]
emb_dim = 50
nb_filters = 16
dropout_rate = 0.1
######################################

if __name__ == '__main__':
    x_train, entity_pos_lst, y_train, vocb, vocb_inv = load_data(unk_limit=unk_limit, max_sent_len=max_sent_len)
    # model = MCCNN(max_sent_len=max_sent_len,
    #             vocb=vocb,
    #             emb_dim=emb_dim,
    #             kernel_lst=kernel_lst,
    #             nb_filters=nb_filters,
    #             dropout_rate=dropout_rate,
    #             optimizer='adam',
    #             lr_rate=learning_rate,
    #             use_pretrained=False)
    model = CNN(max_sent_len=max_sent_len,
                vocb=vocb,
                emb_dim=emb_dim,
                kernel_lst=kernel_lst,
                nb_filters=nb_filters,
                dropout_rate=dropout_rate,
                optimizer='adam',
                non_static=True,
                lr_rate=learning_rate,
                use_pretrained=False)
    model.show_model_summary()
    model.save_model()
    model.train(x_train=x_train, y_train=y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.1)
