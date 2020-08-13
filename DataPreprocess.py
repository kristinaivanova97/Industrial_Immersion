import json

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import h5py
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer

from Class import to_train_val_test_hdf


class DataPreprocess:
    
    def __init__(self):
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        label_list = config['label_list']
        self.label_map = {label: i for i, label in enumerate(label_list)}

        if config['from_bert']:
            self.tokenizer = BertTokenizer.from_pretrained(config['config_of_tokenizer'])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config['config_of_tokenizer'])

        self.file = config['path_train_data_full_conll']
        self.data_dir = config['data_dir_with_full_labels_hdf']

    def process_batch(self):

        with open(self.file, 'r', encoding='utf-8') as file:
            with h5py.File(self.data_dir + 'ids_all.hdf5', 'w') as f:
                dset_input_ids = f.create_dataset("input_ids", (1200000, 512), maxshape=(1500000,512), dtype='i8')
                dset_input_mask = f.create_dataset("input_mask", (1200000, 512), maxshape=(1500000,512), dtype='i1')
                dset_label_ids = f.create_dataset("label_ids", (1200000, 512), maxshape=(1500000,512), dtype='i1')
                line = file.readline()
                stripped_line = line.strip()
                line_list = stripped_line.split()
                i = 0
                pbar = tqdm(total=29247027)
                list_of_words = []
                list_of_labeles = []
                while line:

                    if len(line_list) > 1:
                        list_of_words.append(line_list[0])
                        list_of_labeles.append(line_list[1])
                    else:
                        input_ids, input_mask, label_ids, nopad = self.convert_single_example(sentence=list_of_words,
                                                                                              sentence_labels=list_of_labeles,
                                                                                              part_of_word=False)

                        dset_input_ids[i, :] = input_ids[:]
                        dset_input_mask[i, :] = input_mask[:]
                        dset_label_ids[i, :] = label_ids[:]
                        list_of_words = []
                        list_of_labeles = []
                        i += 1

                    line = file.readline()
                    stripped_line = line.strip()
                    line_list = stripped_line.split()

                    pbar.update(1)
                    if i == 1000:
                        break

                pbar.close()





    def convert_single_example(self, sentence, sentence_labels, max_seq_length = 512, part_of_word = False):

        tokens = []
        labels = []
        nopad = []
        for i, word in enumerate(sentence):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            word_label = sentence_labels[i]
            for m in range(len(token)):
                if part_of_word:
                    if m == 0:
                        labels.append(word_label)
                    else:
                        labels.append("[##]")
                else:
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


def main():

    # data_processor = DataPreprocess()
    # data_processor.process_batch()

    to_train_val_test_hdf(data_dir='./new_data/', output_dir='./data_2/', train_part=0.6, val_part=0.2, test_part=0.2, length=140000, random_seed=1)


if __name__ == "__main__":
    main()

