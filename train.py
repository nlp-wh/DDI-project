from load_data_ddi import load_data, sentence_split_for_pcnn
from model import CNN, MCCNN, BILSTM, PCNN, MC_PCNN
# Get hyperparameter informaction
from config import *

if __name__ == '__main__':
    (tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_pos_tuple_lst, tr_y), \
        (de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_pos_tuple_lst, de_y), \
        (te_sentences2idx, te_d1_pos_lst, te_d2_pos_lst, te_pos_tuple_lst, te_y), \
        (vocb, vocb_inv), (d1_vocb, d2_vocb) = load_data(unk_limit=unk_limit, max_sent_len=max_sent_len, dev_size=dev_size)

    if train_mode.lower() == 'pcnn' or train_mode.lower() == 'mcpcnn':
        (tr_sent_left, tr_d1_left, tr_d2_left), (tr_sent_mid, tr_d1_mid, tr_d2_mid), (tr_sent_right, tr_d1_right, tr_d2_right) = \
            sentence_split_for_pcnn(sentences2idx=tr_sentences2idx, d1_pos_lst=tr_d1_pos_lst, d2_pos_lst=tr_d1_pos_lst,
                                    pos_tuple_lst=tr_pos_tuple_lst, max_sent_len=max_sent_len)

        del tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_pos_tuple_lst

        (de_sent_left, de_d1_left, de_d2_left), (de_sent_mid, de_d1_mid, de_d2_mid), (de_sent_right, de_d1_right, de_d2_right) = \
            sentence_split_for_pcnn(sentences2idx=de_sentences2idx, d1_pos_lst=de_d1_pos_lst, d2_pos_lst=de_d1_pos_lst,
                                    pos_tuple_lst=de_pos_tuple_lst, max_sent_len=max_sent_len)

        del de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_pos_tuple_lst

        (te_sent_left, te_d1_left, te_d2_left), (te_sent_mid, te_d1_mid, te_d2_mid), (te_sent_right, te_d1_right, te_d2_right) = \
            sentence_split_for_pcnn(sentences2idx=te_sentences2idx, d1_pos_lst=te_d1_pos_lst, d2_pos_lst=te_d1_pos_lst,
                                    pos_tuple_lst=te_pos_tuple_lst, max_sent_len=max_sent_len)

        del te_sentences2idx, te_d1_pos_lst, te_d2_pos_lst, te_pos_tuple_lst

        if train_mode.lower() == 'pcnn':
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
                         num_classes=num_classes,
                         hidden_unit_size=hidden_unit_size,
                         use_batch_norm=use_batch_norm,
                         use_l2_reg=use_l2_reg,
                         reg_coef_conv=reg_coef_conv,
                         reg_coef_dense=reg_coef_dense)
        else:
            model = MC_PCNN(max_sent_len=max_sent_len,
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
                            unk_limit=unk_limit,
                            num_classes=num_classes,
                            hidden_unit_size=hidden_unit_size,
                            use_batch_norm=use_batch_norm,
                            use_l2_reg=use_l2_reg,
                            reg_coef_conv=reg_coef_conv,
                            reg_coef_dense=reg_coef_dense)

        model.show_model_summary()
        model.save_model()
        model.train(nb_epoch=nb_epoch, batch_size=batch_size, train_data=(
            (tr_sent_left, tr_d1_left, tr_d2_left), (tr_sent_mid, tr_d1_mid, tr_d2_mid), (tr_sent_right, tr_d1_right, tr_d2_right), tr_y),
            dev_data=(
            (de_sent_left, de_d1_left, de_d2_left), (de_sent_mid, de_d1_mid, de_d2_mid), (de_sent_right, de_d1_right, de_d2_right), de_y))
        model.evaluate(
            test_data=((te_sent_left, te_d1_left, te_d2_left), (te_sent_mid, te_d1_mid, te_d2_mid), (te_sent_right, te_d1_right, te_d2_right), te_y),
            batch_size=batch_size)
    else:
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
                        num_classes=num_classes,
                        hidden_unit_size=hidden_unit_size,
                        use_batch_norm=use_batch_norm,
                        use_l2_reg=use_l2_reg,
                        reg_coef_conv=reg_coef_conv,
                        reg_coef_dense=reg_coef_dense)

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
                          unk_limit=unk_limit,
                          num_classes=num_classes,
                          hidden_unit_size=hidden_unit_size,
                          use_batch_norm=use_batch_norm,
                          use_l2_reg=use_l2_reg,
                          reg_coef_conv=reg_coef_conv,
                          reg_coef_dense=reg_coef_dense)

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
