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
            
    # Create labels for every word (or comma) in text
    def label_dict(self):
        tsya_search = re.compile(r'тся\b')
        tsiya_search = re.compile(r'ться\b')
        dicty = {}
        k = 0
        i = 0
        new_data_x = []
        for raw, label in zip(self.data['x'], self.data['y']):
            m = tsya_search.findall(raw)
            m2 = tsiya_search.findall(raw)
            if label == 1 or len(m) + len(m2) == 1:
                new_data_x.append(raw)
                #for j,word in enumerate(raw.split()):
                for j, word in  enumerate(re.findall(r'\w+|[^\w\s]', raw, re.UNICODE)):
                    # any sequence of letters(+numbers) or comas, points
                    m = tsya_search.findall(word)
                    m2 = tsiya_search.findall(word)
                    dicty.setdefault(i, {})
                    if (len(m) == 1 and label == 1) or (len(m2) == 1 and label == 0):
                        dicty[i][j] = "тся"
                    elif (len(m) == 1 and label == 0) or (len(m2) == 1 and label == 1):
                        dicty[i][j] = "ться"
                    else:
                        dicty[i][j] = "O"
            else:
                k+=1
            i+=1
        print("Num of sentences which have a mistake and contain more than 1 word with ться/тся = ", k)
        return dicty, new_data_x
    
    def process(self):

        with open(my_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            list_of_words = []
            list_of_labeles = []
            # list_of_sentenes = []
            # list_of_labeles_sentence = []
            for line in lines:
                stripped_line = line.strip()
                line_list = stripped_line.split()
                if len(line_list) > 1:
                    list_of_words.append(line_list[0])
                    list_of_labeles.append(line_list[1])
                else:
                    # list_of_sentenes.append(list_of_words)
                    # list_of_labeles_sentence.append(list_of_labeles)


                    input_ids, input_mask, label_ids, nopad = self.convert_single_example(list_of_words, list_of_labeles)
                    self.input_ids.append(input_ids)
                    self.attention_masks.append(input_mask)
                    self.label_ids.append(label_ids)
                    self.nopad.append(nopad)
                    list_of_words = []
                    list_of_labeles = []

        # for k,raw in tqdm(enumerate(self.data)):
        #     input_ids, input_mask,label_ids, nopad = self.convert_single_example(raw, self.y_label[k])
        #     self.input_ids.append(input_ids)
        #     self.attention_masks.append(input_mask)
        #     self.label_ids.append(label_ids)
        #     self.nopad.append(nopad)


    def convert_single_example(self, sentence, sentence_label):
        max_seq_length = 512


        tokens = []
        labels = []
        nopad = []
        for i, word in enumerate(sentence):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            word_label = sentence_label[i]
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





    # find indices for Bert from our data for each sentence
    # def convert_single_example(self, text, y_label, max_seq_length = 512):
    #
    #     #textlist = text.split()
    #     textlist = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    #     tokens = []
    #     labels = []
    #     nopad = []
    #     for i, word in enumerate(textlist):
    #         token = self.tokenizer.tokenize(word)
    #         tokens.extend(token)
    #         label_1 = y_label[i]
    #         for m in range(len(token)):
    #             labels.append(label_1)
    #
    #
    #
    #     if len(tokens) >= max_seq_length - 1:
    #         tokens = tokens[0:(max_seq_length - 2)]
    #         labels = labels[0:(max_seq_length - 2)]
    #     ntokens = []
    #     label_ids = []
    #     ntokens.append("[CLS]")
    #     label_ids.append(self.label_map["[CLS]"])
    #     for i, token in enumerate(tokens):
    #         ntokens.append(token)
    #         label_ids.append(self.label_map[labels[i]])
    #
    #     ntokens.append("[SEP]")
    #
    #
    #     nopad.append(len(ntokens))
    #     label_ids.append(self.label_map["[SEP]"])
    #     input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
    #     input_mask = [1] * len(input_ids)
    #
    #     while len(input_ids) < max_seq_length:
    #         input_ids.append(0)
    #         input_mask.append(0)
    #         label_ids.append(0)
    #         ntokens.append("[Padding]")
    #     assert len(input_ids) == max_seq_length
    #     assert len(input_mask) == max_seq_length
    #     assert len(label_ids) == max_seq_length
    #     return input_ids, input_mask, label_ids, nopad
    #
    def save_labels(self, path):
 
        my_file = open(path, 'w', encoding='utf-8')
        for raw in tqdm(self.y_label):
            for elem in raw:
                my_file.write(str(elem))
                my_file.write(' ')
            my_file.write('\n')
        my_file.close()
        
    def save_indices(self, ftype):
        self.process()
        # save to 3 files
        file_names = ['input_ids_' + ftype + '.txt', 'input_mask_' + ftype + '.txt', 'label_ids_' + ftype + '.txt']
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
    train_processor = DataPreprocess(path_to_file=path_to_train)
    val_processor = DataPreprocess(path_to_file=path_to_val)

    train_processor.save_labels(path=path_to_train_labels)
    val_processor.save_labels(path=path_to_val_labels)

    train_processor.save_indices(ftype='train')
    val_processor.save_indices(ftype='val')


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
