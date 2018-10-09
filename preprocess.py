import os
import sys
import re
from geniatagger import GeniaTagger

# Row setting
# DDI_ID, Drug_1_Name, Drug_1_Drugbankid, ent_type, Drug_2_Name, Drug_2_Drugbankid, Is_DDI, DDI_Type, ent_type, Sentence_Text
# idx: 1, 4, 7, 9

'''
1. Load data from 'data/ddi.tsv
2. Take the specific rows (idx: 1, 4, 7, 9)
3. Replace the drug names into DRUGA and DRUGB
4. Replace '-' into ''
5. sent = re.sub('\d', 'dg', sent) (Still needs to be considered)
6. Tokenize with GeniaTagger
7. Replace DRUGA, DRUGB into the original drug name
8. lowercase
'''

data_dir = 'data'
orig_file_name = 'ddi.tsv'
train_file_name = 'train.tsv'

if __name__ == '__main__':
    tagger = GeniaTagger(os.path.join('geniatagger-3.0.2', 'geniatagger'))
    orig_f = open(os.path.join(data_dir, orig_file_name), 'r', encoding='utf-8')
    train_f = open(os.path.join(data_dir, train_file_name), 'w', encoding='utf-8')
    # 1. Load data from 'data/ddi.tsv
    dataset = []
    for line in orig_f:
        line = line.strip()
        items = line.split('\t')
        # 2. Take the specific rows (idx: 1, 4, 7, 9)
        drug_a = items[1]
        drug_b = items[4]
        rel = items[7]
        sent = items[9]

        # 3. Replace the drug names into DRUGA and DRUGB
        sent = sent.replace(drug_a, "DRUGA").replace(drug_b, "DRUGB")

        # 4. Replace '-' into ''
        sent = sent.replace('-', ' ')
        sent = sent.replace('/', ' / ')

        # 6. Tokenize with GeniaTagger
        tokenize_result = tagger.parse(sent)
        tokens = []
        for token_info in tokenize_result:
            token = token_info[0]
            # if the token is number, change it into dg
            if bool(re.search('^\d+$|^\d+.\d+$', token)):
                token = 'dg'
            tokens.append(token)
        final_sent = ' '.join(tokens)
        # 7. Replace DRUGA, DRUGB into the original drug name (Do this task on data loading time)
        # final_sent = final_sent.replace('DRUGA', drug_a).replace('DRUGB', drug_b)
        # 8. lowercase
        final_sent = final_sent.lower()

        # 9. Save the file into train
        train_f.write('{}\t{}\t{}\t{}\n'.format(drug_a, drug_b, rel, final_sent))
    orig_f.close()
    train_f.close()
