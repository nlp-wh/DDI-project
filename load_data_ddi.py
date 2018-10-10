from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import numpy as np
import os

data_dir = 'data'
word2vec_dir = 'word2vec'
train_filename = 'train.tsv'
vocab_filename = 'vocab'

# Check whether the data file exists
if not os.path.exists(os.path.join(data_dir, train_filename)):
    raise FileNotFoundError("[{}] file not found".format(train_filename))


rel_class = {'advise': 0, 'effect': 1, 'mechanism': 2, 'int': 3}


def load_sentence():
    sentences = []
    drug1_lst = []
    drug2_lst = []
    rel_lst = []

    with open(os.path.join(data_dir, train_filename), 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            item_lst = line.strip().split('\t')
            drug1_lst.append(item_lst[0])
            drug2_lst.append(item_lst[1])
            rel_lst.append(rel_class[item_lst[2]])
            sent = item_lst[3].split(' ')
            for idx, token in enumerate(sent):
                if token.find('druga') != -1:
                    sent[idx] = token.replace('druga', drug1_lst[line_num])
                elif token.find('drugb') != -1:
                    sent[idx] = token.replace('drugb', drug2_lst[line_num])
            sentences.append(sent)
            line_num += 1

    assert len(sentences) == len(drug1_lst) == len(drug2_lst) == len(rel_lst)
    print('sentences[0]:', sentences[0])
    print('drug1_lst[0]:', drug1_lst[0])
    print('drug2_lst[0]:', drug2_lst[0])
    print('rel_lst[0]:', rel_lst[0])
    return sentences, drug1_lst, drug2_lst, rel_lst


def find_drug1_drug2_in_sentence(sentences, drug1_lst, drug2_lst):
    '''
    0:None
    1:Drug1
    2:Drug2
    '''
    entity_pos_lst = []
    for idx, sentence in enumerate(sentences):
        entity_in_sent = []
        drug1 = drug1_lst[idx]
        drug2 = drug2_lst[idx]
        for word in sentence:
            if word == drug1:
                entity_in_sent.append(1)
            elif word == drug2:
                entity_in_sent.append(2)
            else:
                entity_in_sent.append(0)
        # if entity_in_sent.count(1) != 1 or entity_in_sent.count(2) != 1:
        #     print(entity_in_sent)
        #     print(sentences[idx])
        entity_pos_lst.append(entity_in_sent)
    print('entity_pos_lst[0]:', entity_pos_lst[0])

    return entity_pos_lst


def build_word_vocab(sentences):
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


def word2idx(sentences, vocb, unk_limit=10000):
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


def load_word_matrix(vocb, emb_dim=100, unk_limit=10000):
    embedding_index = dict()
    file_name = 'glove.6B.{}d.txt'.format(emb_dim)
    vec_file_name = os.path.join(word2vec_dir, file_name)
    if not os.path.exists(vec_file_name):
        raise FileNotFoundError(vec_file_name + ' not found')
    with open(vec_file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    word_matrix = np.zeros((unk_limit, emb_dim))
    cnt = 0
    for word, i in vocb.items():
        if i < unk_limit:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                word_matrix[i] = embedding_vector
            else:
                word_matrix[i] = np.random.uniform(-1.0, 1.0, emb_dim)
                cnt += 1
    print('{} words not in word vector'.format(cnt))
    return word_matrix


def pad_sequence(seq, max_sent_len=50):
    return pad_sequences(seq, padding='post', maxlen=max_sent_len)


def one_hot_encoding(rel_lst):
    return keras.utils.to_categorical(rel_lst, num_classes=len(rel_class))


def load_data(unk_limit=5000, max_sent_len=50):
    sentences, drug1_lst, drug2_lst, rel_lst = load_sentence()
    entity_pos_lst = find_drug1_drug2_in_sentence(sentences, drug1_lst, drug2_lst)
    entity_pos_lst = pad_sequence(entity_pos_lst, max_sent_len=max_sent_len)
    vocb, vocb_inv = build_word_vocab(sentences)
    sentences2idx = word2idx(sentences, vocb, unk_limit=unk_limit)
    sentences2idx = pad_sequence(sentences2idx, max_sent_len=max_sent_len)
    y = one_hot_encoding(rel_lst)
    print('y[0]:', y[0])
    return (sentences2idx, entity_pos_lst, y), (vocb, vocb_inv), (sentences, drug1_lst, drug2_lst, rel_lst)


if __name__ == '__main__':
    load_data()
