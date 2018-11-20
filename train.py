from load_data_ddi import load_data, sentence_split_for_pcnn
from model import CNN, MCCNN, BILSTM, PCNN, MC_PCNN, MC_PCNN_ATT
from config import Config  # Get hyperparameter informaction
import numpy as np

if __name__ == '__main__':
    cfg = Config()
    (tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_pos_tuple_lst, tr_y), \
    (de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_pos_tuple_lst, de_y), \
    (te_sentences2idx, te_d1_pos_lst, te_d2_pos_lst, te_pos_tuple_lst, te_y), \
    (vocb, vocb_inv), (d1_vocb, d2_vocb) = load_data(unk_limit=cfg.unk_limit, max_sent_len=cfg.max_sent_len, dev_size=cfg.dev_size)

    # Error has happened on old keras version
    tr_sentences2idx = np.asarray(tr_sentences2idx)
    tr_d1_pos_lst = np.asarray(tr_d1_pos_lst)
    tr_d2_pos_lst = np.asarray(tr_d2_pos_lst)
    tr_pos_tuple_lst = np.asarray(tr_pos_tuple_lst)
    tr_y = np.asarray(tr_y)
    de_sentences2idx = np.asarray(de_sentences2idx)
    de_d1_pos_lst = np.asarray(de_d1_pos_lst)
    de_d2_pos_lst = np.asarray(de_d2_pos_lst)
    de_pos_tuple_lst = np.asarray(de_pos_tuple_lst)
    de_y = np.asarray(de_y)
    te_sentences2idx = np.asarray(te_sentences2idx)
    te_d1_pos_lst = np.asarray(te_d1_pos_lst)
    te_d2_pos_lst = np.asarray(te_d2_pos_lst)
    te_pos_tuple_lst = np.asarray(te_pos_tuple_lst)
    te_y = np.asarray(te_y)

    if cfg.train_mode.lower() == 'pcnn' or cfg.train_mode.lower() == 'mcpcnn' or cfg.train_mode.lower() == 'mcpcnn_att':
        (tr_sent_left, tr_d1_left, tr_d2_left), (tr_sent_mid, tr_d1_mid, tr_d2_mid), (tr_sent_right, tr_d1_right, tr_d2_right) = \
            sentence_split_for_pcnn(sentences2idx=tr_sentences2idx, d1_pos_lst=tr_d1_pos_lst, d2_pos_lst=tr_d1_pos_lst,
                                    pos_tuple_lst=tr_pos_tuple_lst, max_sent_len=cfg.max_sent_len)

        del tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_pos_tuple_lst

        (de_sent_left, de_d1_left, de_d2_left), (de_sent_mid, de_d1_mid, de_d2_mid), (de_sent_right, de_d1_right, de_d2_right) = \
            sentence_split_for_pcnn(sentences2idx=de_sentences2idx, d1_pos_lst=de_d1_pos_lst, d2_pos_lst=de_d1_pos_lst,
                                    pos_tuple_lst=de_pos_tuple_lst, max_sent_len=cfg.max_sent_len)

        del de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_pos_tuple_lst

        (te_sent_left, te_d1_left, te_d2_left), (te_sent_mid, te_d1_mid, te_d2_mid), (te_sent_right, te_d1_right, te_d2_right) = \
            sentence_split_for_pcnn(sentences2idx=te_sentences2idx, d1_pos_lst=te_d1_pos_lst, d2_pos_lst=te_d1_pos_lst,
                                    pos_tuple_lst=te_pos_tuple_lst, max_sent_len=cfg.max_sent_len)

        del te_sentences2idx, te_d1_pos_lst, te_d2_pos_lst, te_pos_tuple_lst

        if cfg.train_mode.lower() == 'pcnn':
            model = PCNN(cfg, vocb, d1_vocb, d2_vocb)
        elif cfg.train_mode.lower() == 'mcpcnn':
            model = MC_PCNN(cfg, vocb, d1_vocb, d2_vocb)
        elif cfg.train_mode.lower() == 'mcpcnn_att':
            model = MC_PCNN_ATT(cfg, vocb, d1_vocb, d2_vocb)

        model.show_model_summary()
        model.save_model()
        model.train(
            train_data=((tr_sent_left, tr_d1_left, tr_d2_left), (tr_sent_mid, tr_d1_mid, tr_d2_mid), (tr_sent_right, tr_d1_right, tr_d2_right), tr_y),
            dev_data=((de_sent_left, de_d1_left, de_d2_left), (de_sent_mid, de_d1_mid, de_d2_mid), (de_sent_right, de_d1_right, de_d2_right), de_y))
        model.evaluate(
            test_data=((te_sent_left, te_d1_left, te_d2_left), (te_sent_mid, te_d1_mid, te_d2_mid), (te_sent_right, te_d1_right, te_d2_right), te_y))
    else:
        if cfg.train_mode.lower() == 'cnn':
            model = CNN(cfg, vocb, d1_vocb, d2_vocb)

        elif cfg.train_mode.lower() == 'mccnn':
            model = MCCNN(cfg, vocb, d1_vocb, d2_vocb)

        elif cfg.train_mode.lower() == 'rnn':
            model = BILSTM(cfg, vocb, d1_vocb, d2_vocb)
        else:
            raise Exception("Wrong Training Model")
        model.show_model_summary()
        model.save_model()
        model.train(train_data=(tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_y),
                    dev_data=(de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_y))
        model.evaluate(test_data=((te_sentences2idx, te_d1_pos_lst, te_d2_pos_lst), te_y))
