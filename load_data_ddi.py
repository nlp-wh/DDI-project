from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import numpy as np
import os

data_dir = 'data'
train_filename = 'train'
vocab_filename = 'vocab'


def load_sentence():
    sentences = []
    drugs = []
    effects = []

    with open(os.path.join(data_dir, train_filename), 'r', encoding='utf-8') as f:
        for line in f:
            item_lst = line.rstrip().split('\t')
            sentences.append(item_lst[0].split(' '))
            drugs.append(item_lst[1])
            effects.append(item_lst[2])

    assert len(sentences) == len(drugs) == len(effects)
    print('sentences[0]:', sentences[0])
    print('drugs[0]:', drugs[0])
    print('effects[0]:', effects[0])
    return sentences, drugs, effects


def find_drug_effect_in_sentence(sentences, drugs, effects):
    '''
    0:None
    1:Drug
    2:Effect
    '''
    entity_pos_lst = []
    for idx, sentence in enumerate(sentences):
        entity_in_sent = []
        drug = drugs[idx]
        effect = effects[idx]
        for word in sentence:
            if word == drug:
                entity_in_sent.append(1)
            elif word == effect:
                entity_in_sent.append(2)
            else:
                entity_in_sent.append(0)
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
    with open(os.path.join(data_dir, 'glove.6B.{}d.txt'.format(emb_dim)), 'r', encoding='utf-8') as f:
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
                word_matrix[i] = np.random.uniform(-1, 1, emb_dim)
                cnt += 1
    print('{} words not in glove'.format(cnt))
    return word_matrix


def pad_sequence(sentences2idx, max_sent_len=50):
    return pad_sequences(sentences2idx, padding='post', maxlen=max_sent_len)


def load_data(unk_limit=10000, max_sent_len=50):
    sentences, drugs, effects = load_sentence()
    entity_pos_lst = find_drug_effect_in_sentence(sentences, drugs, effects)
    vocb, vocb_inv = build_word_vocab(sentences)
    sentences2idx = word2idx(sentences, vocb, unk_limit=unk_limit)
    sentences2idx = pad_sequence(sentences2idx, max_sent_len=max_sent_len)
    y = [1] * len(sentences)
    y = np.asarray(y)

    return sentences2idx, entity_pos_lst, y, vocb, vocb_inv


if __name__ == '__main__':
    load_data()
