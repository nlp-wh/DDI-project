import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

from collections import Counter
import numpy as np
import os

data_dir = 'data'
word2vec_dir = 'word2vec'
train_filename = 'train.tsv'
test_filename = 'test.tsv'
vocab_filename = 'vocab'

word_vec_file_lst = [
    'pmc',
    'pubmed',
    'pubmed_and_pmc',
    'pubmed_myself',
    'wiki_pubmed'
]

# Check whether the data file exists
if not os.path.exists(os.path.join(data_dir, train_filename)):
    raise FileNotFoundError("[{}] file not found".format(train_filename))

rel_class = {'false': 0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}

'''
Column
[ddi_id, drug_1_name, drug_1_type, drug_1_offset, drug_2_name, drug_2_type, drug_2_offset, ddi_type, sentence]
-> [drug_1_name, drug_2_name, ddi_type, sentence] = [1, 4, 7, 8]
'''


def load_sentence(filename, max_sent_len):
    sentences = []
    drug1_lst = []
    drug2_lst = []
    d1_pos_lst = []
    d2_pos_lst = []
    rel_lst = []
    pos_tuple_lst = []

    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            item_lst = line.strip().split('\t')
            drug1_lst.append(item_lst[1])
            drug2_lst.append(item_lst[4])
            rel_lst.append(rel_class[item_lst[7]])
            sent = item_lst[8].split(' ')
            # Before Replacing, build positing list!!
            d1, d2, e_pos_tup = build_position_embedding(sent, max_sent_len)
            d1_pos_lst.append(d1)
            d2_pos_lst.append(d2)
            pos_tuple_lst.append(e_pos_tup)
            # Then replace drug name
            '''
            for idx, token in enumerate(sent):
                if token.find('druga') != -1:
                    sent[idx] = token.replace('druga', drug1_lst[line_num])
                elif token.find('drugb') != -1:
                    sent[idx] = token.replace('drugb', drug2_lst[line_num])
            '''
            sentences.append(sent)
            line_num += 1

    assert len(sentences) == len(drug1_lst) == len(drug2_lst) == len(rel_lst) == len(d1_pos_lst) == len(d2_pos_lst) == len(pos_tuple_lst)
    # Testing
    print('sentences[0]:', sentences[0])
    print('drug1_lst[0]:', drug1_lst[0])
    print('drug2_lst[0]:', drug2_lst[0])
    print('rel_lst[0]:', rel_lst[0])
    # print('d1_pos_lst[0]:', d1_pos_lst[0])
    # print('d2_pos_lst[0]:', d2_pos_lst[0])
    print('pos_tuple_lst[0]:', pos_tuple_lst[0])
    return sentences, drug1_lst, drug2_lst, rel_lst, d1_pos_lst, d2_pos_lst, pos_tuple_lst


def load_test_pair_id():
    pair_id_lst = []
    with open(os.path.join(data_dir, test_filename), 'r', encoding='utf-8') as f:
        for line in f:
            pair_id_lst.append(line.split('\t')[0])
    return pair_id_lst


def build_position_embedding(sent_list, max_sent_len):
    # print(sent_list)
    e1 = sent_list.index('druga')
    e2 = sent_list.index('drugb')
    # distance1 feature
    d1 = []
    for i in range(max_sent_len):
        if i < e1:
            d1.append(str(i - e1))
        elif i > e1:
            d1.append(str(i - e1))
        else:
            d1.append('0')
    # distance2 feature
    d2 = []
    for i in range(max_sent_len):
        if i < e2:
            d2.append(str(i - e2))
        elif i > e2:
            d2.append(str(i - e2))
        else:
            d2.append('0')
    return d1, d2, [e1, e2]


def build_position_vocab(pos_lst):
    # input: [tr_d1_pos_lst, te_d1_pos_lst]
    # d1, d2 각각 따로 함수 호출해야 함
    sent_list = sum(pos_lst, [])
    wf = {}
    for sent in sent_list:
        for w in sent:
            if w in wf:
                wf[w] += 1
            else:
                wf[w] = 0

    wl = []
    for w, f in wf.items():
        wl.append(w)
    # Append <PAD>
    wl.insert(0, '<PAD>')
    return wl


def map_word_to_id(sent_contents, word_list):
    T = []
    for sent in sent_contents:
        t = []
        for w in sent:
            t.append(word_list.index(w))
        T.append(t)
    return T


def build_word_vocab(sentences):
    '''
    if os.path.exists(os.path.join(data_dir, vocab_filename)):
        vocb = {}
        vocb_inv = {}
        with open(os.path.join(data_dir, vocab_filename), 'r', encoding='utf-8') as f:
            for line in f:
                idx, w = line.rstrip().split('\t')
                idx = int(idx)
                vocb[w] = idx
                vocb_inv[idx] = w
    else:
    '''
    words = []
    for sentence in sentences:
        words.extend(sentence)

    word_counts = Counter(words)
    vocb_lst = [x[0] for x in word_counts.most_common()]

    vocb = dict()
    vocb['<PAD>'] = 0
    vocb['<GO>'] = 1
    vocb['<UNK>'] = 2
    for idx, w in enumerate(vocb_lst):
        vocb[w] = idx + 3
    print('vocb_len', len(vocb))
    print('vocb_lst[0]', vocb_lst[0])
    print('vocb_lst[1]', vocb_lst[1])
    vocb_inv = {idx: w for w, idx in vocb.items()}
    # Save vocb
    with open(os.path.join(data_dir, vocab_filename), 'w', encoding='utf-8') as f:
        for w, idx in vocb.items():
            f.write("{}\t{}\n".format(idx, w))
    return vocb, vocb_inv


def word2idx(sentences, vocb, unk_limit):
    sentences2idx = []

    for sentence in sentences:
        w_id = []
        for word in sentence:
            word_idx = vocb.get(word, vocb['<UNK>'])
            if word_idx >= unk_limit:
                word_idx = vocb['<UNK>']
            w_id.append(word_idx)
        sentences2idx.append(w_id)
    assert len(sentences) == len(sentences2idx)

    return sentences2idx


def load_word_matrix(vocb, emb_dim, unk_limit):
    print("Loading word2vec...")
    # load word2vec by gensim library
    # Dimension is 200
    file_name = 'PubMed-and-PMC-w2v.bin'
    vec_file_name = os.path.join(word2vec_dir, file_name)
    if not os.path.exists(vec_file_name):
        raise FileNotFoundError(vec_file_name + ' not found')
    model = KeyedVectors.load_word2vec_format(vec_file_name, binary=True)
    if len(vocb) < unk_limit:
        word_matrix = np.zeros((len(vocb), emb_dim))
    else:
        word_matrix = np.zeros((unk_limit, emb_dim))
    print('word_matrix.shape:', word_matrix.shape)
    cnt = 0
    for word, i in vocb.items():
        if i < unk_limit:
            try:
                if word == 'druga' or word == 'drugb' or word == 'drugn':
                    vector = model['drug']
                else:
                    vector = model[word]
                word_matrix[i] = np.asarray(vector)
            except:
                # word2vec에 없는 단어일 시
                word_matrix[i] = np.random.uniform(-1.0, 1.0, emb_dim)
                cnt += 1
            '''
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                word_matrix[i] = embedding_vector
            else:
                word_matrix[i] = np.random.uniform(-1.0, 1.0, emb_dim)
                cnt += 1
            '''
    print('{} words not in word vector'.format(cnt))
    return word_matrix


def load_word_matrix_from_txt(vocb, emb_dim, unk_limit, word_matrix_file_name):
    print("Loading word2vec {}...".format(word_matrix_file_name))
    # load word2vec by gensim library
    # Dimension is 200
    vec_file_name = os.path.join(word2vec_dir, word_matrix_file_name)
    if not os.path.exists(vec_file_name):
        raise FileNotFoundError(vec_file_name + ' not found')
    model = dict()
    with open(vec_file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            values = line.split()
            word = values[0].lower()  # make it lowercase
            coefs = np.asarray(values[1:], dtype='float32')
            model[word] = coefs
    # model = KeyedVectors.load_word2vec_format(vec_file_name, binary=True)
    if len(vocb) < unk_limit:
        word_matrix = np.zeros((len(vocb), emb_dim))
    else:
        word_matrix = np.zeros((unk_limit, emb_dim))
    cnt = 0
    for word, i in vocb.items():
        if i < unk_limit:
            try:
                if word == 'druga' or word == 'drugb' or word == 'drugn':
                    vector = model['drug']
                else:
                    vector = model[word]
                word_matrix[i] = np.asarray(vector)
            except:
                # word2vec에 없는 단어일 시
                word_matrix[i] = np.random.uniform(-0.25, 0.25, emb_dim)
                cnt += 1
            '''
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                word_matrix[i] = embedding_vector
            else:
                word_matrix[i] = np.random.uniform(-1.0, 1.0, emb_dim)
                cnt += 1
            '''
    print('{} words not in word vector'.format(cnt))
    # make zero padding array to 0
    word_matrix[0] = np.full(shape=emb_dim, fill_value=0.0, dtype=np.float32)
    print('word_matrix.shape:', word_matrix.shape)
    return word_matrix


def load_word_matrix_all(vocb, emb_dim, unk_limit):
    word_matrix_lst = []
    for w2v_filename in word_vec_file_lst:
        word_matrix_lst.append(load_word_matrix_from_txt(vocb, emb_dim, unk_limit, w2v_filename))
    return word_matrix_lst


def pad_sequence(seq, max_sent_len):
    return pad_sequences(seq, padding='post', maxlen=max_sent_len)


def one_hot_encoding(rel_lst):
    return keras.utils.to_categorical(rel_lst, num_classes=len(rel_class))


def train_dev_split(sentence, d1_pos_lst, d2_pos_lst, pos_tuple_lst, y, dev_size=0.1, shuffle=True):
    zip_x = []
    for s, d1, d2, pos_tuple in zip(sentence, d1_pos_lst, d2_pos_lst, pos_tuple_lst):
        zip_x.append((s, d1, d2, pos_tuple))

    X_train, X_dev, tr_y, de_y = train_test_split(zip_x, y, test_size=dev_size, shuffle=shuffle)
    tr_sentence = []
    tr_d1_pos_lst = []
    tr_d2_pos_lst = []
    tr_pos_tuple_lst = []
    de_sentence = []
    de_d1_pos_lst = []
    de_d2_pos_lst = []
    de_pos_tuple_lst = []
    for item in X_train:
        tr_sentence.append(item[0])
        tr_d1_pos_lst.append(item[1])
        tr_d2_pos_lst.append(item[2])
        tr_pos_tuple_lst.append(item[3])
    for item in X_dev:
        de_sentence.append(item[0])
        de_d1_pos_lst.append(item[1])
        de_d2_pos_lst.append(item[2])
        de_pos_tuple_lst.append(item[3])

    assert len(tr_sentence) == len(tr_d1_pos_lst) == len(tr_d2_pos_lst) == len(tr_pos_tuple_lst) == len(tr_y)
    assert len(de_sentence) == len(de_d1_pos_lst) == len(de_d2_pos_lst) == len(de_pos_tuple_lst) == len(de_y)

    return (tr_sentence, tr_d1_pos_lst, tr_d2_pos_lst, tr_pos_tuple_lst, tr_y), (de_sentence, de_d1_pos_lst, de_d1_pos_lst, de_pos_tuple_lst, de_y)


def load_data(unk_limit, max_sent_len, dev_size):
    tr_sentences, tr_drug1_lst, tr_drug2_lst, tr_rel_lst, tr_d1_pos_lst, tr_d2_pos_lst, tr_pos_tuple_lst = load_sentence(train_filename, max_sent_len)
    te_sentences, te_drug1_lst, te_drug2_lst, te_rel_lst, te_d1_pos_lst, te_d2_pos_lst, te_pos_tuple_lst = load_sentence(test_filename, max_sent_len)

    # Build distance position vocb
    d1_vocb = build_position_vocab([tr_d1_pos_lst, te_d1_pos_lst])
    d2_vocb = build_position_vocab([tr_d2_pos_lst, te_d2_pos_lst])

    # position to idx
    tr_d1_pos_lst = map_word_to_id(tr_d1_pos_lst, d1_vocb)
    tr_d2_pos_lst = map_word_to_id(tr_d2_pos_lst, d2_vocb)
    te_d1_pos_lst = map_word_to_id(te_d1_pos_lst, d1_vocb)
    te_d2_pos_lst = map_word_to_id(te_d2_pos_lst, d2_vocb)

    # print("tr_d1_pos_lst[0]:", tr_d1_pos_lst[0])
    # print("tr_d2_pos_lst[0]:", tr_d2_pos_lst[0])
    # print("te_d1_pos_lst[0]:", te_d1_pos_lst[0])
    # print("te_d2_pos_lst[0]:", te_d2_pos_lst[0])

    # Build vocab only with train data
    vocb, vocb_inv = build_word_vocab(tr_sentences)

    # Word to idx
    tr_sentences2idx = word2idx(tr_sentences, vocb, unk_limit=unk_limit)
    te_sentences2idx = word2idx(te_sentences, vocb, unk_limit=unk_limit)

    # Padding
    tr_sentences2idx = pad_sequence(tr_sentences2idx, max_sent_len=max_sent_len)
    te_sentences2idx = pad_sequence(te_sentences2idx, max_sent_len=max_sent_len)

    # numpy to list
    tr_sentences2idx = tr_sentences2idx.tolist()
    te_sentences2idx = te_sentences2idx.tolist()

    # tr_y, te_y
    tr_y = one_hot_encoding(tr_rel_lst)
    te_y = one_hot_encoding(te_rel_lst)
    print('tr_y[0]:', tr_y[0])
    print('te_y[0]:', te_y[0])

    # Train, Dev split
    # Add position tuple for PiecewiseCNN
    (tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_pos_tuple_lst, tr_y), (
        de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_pos_tuple_lst, de_y) = train_dev_split(tr_sentences2idx,
                                                                                                  tr_d1_pos_lst,
                                                                                                  tr_d2_pos_lst,
                                                                                                  tr_pos_tuple_lst,
                                                                                                  tr_y,
                                                                                                  dev_size=dev_size,
                                                                                                  shuffle=True)

    return (tr_sentences2idx, tr_d1_pos_lst, tr_d2_pos_lst, tr_pos_tuple_lst, tr_y), \
           (de_sentences2idx, de_d1_pos_lst, de_d2_pos_lst, de_pos_tuple_lst, de_y), \
           (te_sentences2idx, te_d1_pos_lst, te_d2_pos_lst, te_pos_tuple_lst, te_y), \
           (vocb, vocb_inv), (d1_vocb, d2_vocb)


def to_piece(sequence, pos_tuple_lst):
    left = []
    mid = []
    right = []
    assert len(sequence) == len(pos_tuple_lst)
    for i in range(len(sequence)):
        left_idx = pos_tuple_lst[i][0]
        right_idx = pos_tuple_lst[i][1]
        left.append(sequence[i][0:left_idx + 1])
        mid.append(sequence[i][left_idx + 1:right_idx + 1])
        right.append(sequence[i][right_idx + 1:])
    # Testing
    # print("left[0]:", left[0])
    # print("mid[0]:", mid[0])
    # print("right[0]:", right[0])

    return left, mid, right


def sentence_split_for_pcnn(sentences2idx, d1_pos_lst, d2_pos_lst, pos_tuple_lst, max_sent_len):
    '''
    Return: 
        (sent_left, d1_left, d2_left), 
        (sent_mid, d1_mid, d2_mid), 
        (sent_right, d1_right, d2_right)
    '''
    # Split into 3 parts
    sent_left, sent_mid, sent_right = to_piece(sentences2idx, pos_tuple_lst)
    d1_left, d1_mid, d1_right = to_piece(d1_pos_lst, pos_tuple_lst)
    d2_left, d2_mid, d2_right = to_piece(d2_pos_lst, pos_tuple_lst)

    # Pad sequencing
    sent_left = pad_sequence(sent_left, max_sent_len)
    sent_mid = pad_sequence(sent_mid, max_sent_len)
    sent_right = pad_sequence(sent_right, max_sent_len)

    d1_left = pad_sequence(d1_left, max_sent_len)
    d1_mid = pad_sequence(d1_mid, max_sent_len)
    d1_right = pad_sequence(d1_right, max_sent_len)

    d2_left = pad_sequence(d2_left, max_sent_len)
    d2_mid = pad_sequence(d2_mid, max_sent_len)
    d2_right = pad_sequence(d2_right, max_sent_len)

    return (sent_left, d1_left, d2_left), (sent_mid, d1_mid, d2_mid), (sent_right, d1_right, d2_right)


if __name__ == '__main__':
    load_data(unk_limit=8000, max_sent_len=50, dev_size=0.1)
