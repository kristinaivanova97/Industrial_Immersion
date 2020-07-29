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

max_seq_length = 512
data_dir = "./new_data/"
path_to_data = "./dataset.txt"

class DataPreprocess:
    
    def __init__(self, path_to_file):

        label_list = ["[Padding]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya", "REPLACE_tsya",
                      "[##]"]
        self.label_map = {}
        for (i, label) in enumerate(label_list):
            self.label_map[label] = i
            
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        self.file =  path_to_file


    def generate_batch(self, dataset, number=5000):
        batch = []
        for sample in dataset:
            batch.append(sample)

            if len(batch) == number:
                yield batch
                batch = []


    def write_to_file(self, path):
        datast_gen = self.generate_batch()

        for batch in datast_gen:
            with open(path, "a") as inf:
                inf.writelines(batch)
    '''
    def _process_batch(self):
        
        input_ids_batch = []
        attention_masks_batch = []
        label_ids_batch = []
        nopad_batch = []
        

        with open(self.file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            list_of_words = []
            list_of_labeles = []
            for line in tqdm(lines):
                stripped_line = line.strip()
                line_list = stripped_line.split()
                if len(line_list) > 1:
                    list_of_words.append(line_list[0])
                    list_of_labeles.append(line_list[1])
                else:

                    #input_ids, input_mask, label_ids, nopad = self.convert_single_example_with_part_of_word(list_of_words, list_of_labeles)
                    input_ids, input_mask, label_ids, nopad = self.convert_single_example(list_of_words, list_of_labeles)
                    input_ids_batch.append(input_ids)
                    attention_masks_batch.append(input_mask)
                    label_ids_batch.append(label_ids)
                    nopad_batch.append(nopad)
                    list_of_words = []
                    list_of_labeles = []
                    if len(input_ids_batch) == 5000:
                        yield input_ids_batch, attention_masks_batch, label_ids_batch, nopad_batch
                        
                        input_ids_batch = []
                        attention_masks_batch = []
                        label_ids_batch = []
                        nopad_batch = []
    '''
    def _process(self):

        input_ids_full = []
        attention_masks_full = []
        label_ids_full = []
        nopad_full = []


        with open(self.file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            list_of_words = []
            list_of_labeles = []
            for line in tqdm(lines):
                stripped_line = line.strip()
                line_list = stripped_line.split()
                if len(line_list) > 1:
                    list_of_words.append(line_list[0])
                    list_of_labeles.append(line_list[1])
                else:

                    #input_ids, input_mask, label_ids, nopad = self.convert_single_example_with_part_of_word(list_of_words, list_of_labeles)
                    input_ids, input_mask, label_ids, nopad = self.convert_single_example(list_of_words, list_of_labeles)
                    input_ids_full.append(input_ids)
                    attention_masks_full.append(input_mask)
                    label_ids_full.append(label_ids)
                    nopad_full.append(nopad)
                    list_of_words = []
                    list_of_labeles = []
        return input_ids_full, attention_masks_full, label_ids_full


    def convert_single_example(self, sentence, sentence_labels, max_seq_length = 512):

        tokens = []
        labels = []
        nopad = []
        for i, word in enumerate(sentence):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            word_label = sentence_labels[i]
            for m in range(len(token)):
                labels.append(word_label)

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
    
    def convert_single_example_with_part_of_word(self, sentence, sentence_labels, max_seq_length = 512):

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
        
        #generator = self._process_batch()
        
        input_ids, attention_masks, label_ids = self._process()
        # save to 3 files
        file_names = [data_dir + 'input_ids_' + ftype + '.txt', data_dir + 'input_mask_' + ftype + '.txt', data_dir + 'label_ids_' + ftype + '.txt']
        features = [input_ids, attention_masks, label_ids]
                      
        for j in range(len(file_names)):
            my_file = open (file_names[j], 'w', encoding='utf-8')
            for raw in features[j]:
                for elem in raw:
                    my_file.write(str(elem))
                    my_file.write(' ')
                my_file.write('\n')
            my_file.close()
    
    
    def save_indices_hdf(self, data_dir, train_size = 150000):

        input_ids, attention_masks, label_ids = self._process()
        input_ids = np.array(input_ids)
        attention_masks = np.array(input_mask)
        label_ids = np.array(label_ids)
        val_size = train_size:8
        
        with h5py.File(data_dir + 'ids_all_train.hdf5', 'w') as f:
            dset_input_ids = f.create_dataset("input_ids", (train_size, 512), dtype='i8')
            dset_input_ids[:,:] = input_ids[:train_size, :]
            dset_input_mask = f.create_dataset("input_mask", (train_size, 512), dtype='i1')
            dset_input_mask[:,:] = input_mask[:train_size, :]
            dset_label_ids = f.create_dataset("label_ids", (train_size, 512), dtype='i1')
            dset_label_ids[:,:] = label_ids[:train_size, :]
            f.close()
        with h5py.File(data_dir + 'ids_all_val.hdf5', 'w') as f:
            dset_input_ids = f.create_dataset("input_ids", (val_size, 512), dtype='i8')
            dset_input_ids[:,:] = input_ids[train_size:train_size+val_size, :]
            dset_input_mask = f.create_dataset("input_mask", (val_size, 512), dtype='i1')
            dset_input_mask[:,:] = input_mask[train_size:train_size+val_size, :]
            dset_label_ids = f.create_dataset("label_ids", (val_size, 512), dtype='i1')
            dset_label_ids[:,:] = label_ids[train_size:train_size+val_size, :]
            f.close()

def main():
    data_processor = DataPreprocess(path_to_file=path_to_data)

    #data_processor.save_indices(ftype='data', data_dir = data_dir)
    #data_processor.save_indices(ftype='full_labels', data_dir = data_dir)
    data_processor.save_indices_hdf(data_dir = data_dir)

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
