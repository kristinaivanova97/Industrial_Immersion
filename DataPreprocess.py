import numpy as np
import pandas as pd
import re
from transformers import BertTokenizer
from tqdm import tqdm
import math

path_to_train = '../../orpho/data/tsya_data/train_data_bklif.csv'
path_to_val = '../../orpho/data/tsya_data/val_data_bklif.csv'
path_to_train_labels = 'Labels.txt'
path_to_val_labels = 'Val_labels.txt'

label_list = ["[Padding]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya", "REPLACE_tsya", "[##]"]
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
max_seq_length = 512
data_dir = "./new_data/"
path_to_data = "./dataset.txt"


my_file = 'test.txt'
class DataPreprocess:
    
    def __init__(self, path_to_file):

        label_list = ["[Padding]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya", "REPLACE_tsya",
                      "[##]"]
        self.label_map = {}
        for (i, label) in enumerate(label_list):
            self.label_map[label] = i
            
        self.input_ids = []
        self.attention_masks = []
        self.label_ids = []
        self.nopad = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        self.file = path_to_file

    def _process(self):

        with open(self.file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            list_of_words = []
            list_of_labeles = []

            for line in lines:
                stripped_line = line.strip()
                line_list = stripped_line.split()
                if len(line_list) > 1:
                    list_of_words.append(line_list[0])
                    list_of_labeles.append(line_list[1])
                else:
                    input_ids, input_mask, label_ids, nopad = self.convert_single_example(list_of_words, list_of_labeles)
                    self.input_ids.append(input_ids)
                    self.attention_masks.append(input_mask)
                    self.label_ids.append(label_ids)
                    self.nopad.append(nopad)
                    list_of_words = []
                    list_of_labeles = []

    def convert_single_example(self, sentence, sentence_labels, max_seq_length = 512):

        tokens = []
        labels = []
        nopad = []
        for i, word in enumerate(sentence):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            word_label = sentence_labels[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(word_label)
                else:
                    labels.append("[##]")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        label_ids = []
        ntokens.append("[CLS]")
        label_ids.append(self.label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            label_ids.append(self.label_map[labels[i]])

        ntokens.append("[SEP]")
        nopad.append(len(ntokens))
        label_ids.append(self.label_map["[SEP]"])
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(0)
            ntokens.append("[Padding]")
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        return input_ids, input_mask, label_ids, nopad

    def save_labels(self, path):
 
        my_file = open(path, 'w', encoding='utf-8')
        for raw in tqdm(self.y_label):
            for elem in raw:
                my_file.write(str(elem))
                my_file.write(' ')
            my_file.write('\n')
        my_file.close()
        
    def save_indices(self, ftype, data_dir):
        self._process()
        # save to 3 files
        file_names = [data_dir + 'input_ids_' + ftype + '.txt', data_dir + 'input_mask_' + ftype + '.txt', data_dir + 'label_ids_' + ftype + '.txt']
        features = [self.input_ids, self.attention_masks, self.label_ids]

        for j in range(len(file_names)):
            my_file = open (file_names[j], 'w', encoding='utf-8')
            for raw in features[j]:
                for elem in raw:
                    my_file.write(str(elem))
                    my_file.write(' ')
                my_file.write('\n')
            my_file.close()


def main():
    data_processor = DataPreprocess(path_to_file=path_to_data)

    data_processor.save_indices(ftype='data', data_dir = data_dir)


if __name__ == "__main__":
    main()



"""
# To check the max length of sentences in text
# For every sentence...
max_len = 0
new = TrainProcessor.data['x'].str.strip()
for sent in new:
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)
"""
