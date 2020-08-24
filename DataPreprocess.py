import os

import random

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import h5py
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer



data_dir = "./new_data_pow/"


class DataPreprocess:
    
    def __init__(self, path_to_file):

        #label_list = ["[PAD]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya", "REPLACE_tsya", "[##]"]
        label_list = ["[PAD]", "O", "[##]", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya", "REPLACE_tsya"]
        self.label_map = {label: i for i, label in enumerate(label_list)}

        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.file = path_to_file


    def process_batch(self):

        with open(self.file, 'r', encoding='utf-8') as file:
            with h5py.File(data_dir + 'ids_all.hdf5', 'w') as f:
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
                                                                                              part_of_word=True)

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
                        #labels.append("[PAD]")
                else:
                    labels.append(word_label)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        label_ids = []
        ntokens.append("[CLS]")
        input_mask = []
        input_mask.append(1)
        #label_ids.append(self.label_map["[CLS]"])
        label_ids.append(0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            '''
            if labels[i] == "[##]":
                label_ids.append(-100)
            else:
                label_ids.append(self.label_map[labels[i]])
            '''
            label_ids.append(self.label_map[labels[i]])
            input_mask.append(1)

        ntokens.append("[SEP]")
        input_mask.append(1)
        nopad.append(len(ntokens))
        #label_ids.append(self.label_map["[SEP]"])
        label_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        #input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(0) #ignore index
            ntokens.append("[PAD]")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        return input_ids, input_mask, label_ids, nopad


def to_train_val_test_hdf(data_dir = './new_data/', output_dir = './data/', train_part = 0.6,
                          val_part = 0.2, test_part = 0.2, length = 10000, random_seed = 1):


    if not data_dir:
        data_dir = './new_data/'

    if not output_dir:
        output_dir = './data/'

    parts = ["train", "val", "test"]

    with h5py.File(os.path.join(data_dir, "ids_all.hdf5"), 'r') as f:

            input_data = f['input_ids']

            idxs = list(range(len(input_data)))

            random.seed(random_seed)
            random.shuffle(idxs)

            # counter = 0

            points = (
                int(train_part * length),
                int(train_part * length + val_part * length),
                length
            )

            for params in zip(parts, (0,) + points[:-1], points):

                part, start, end = params
                with h5py.File(os.path.join(output_dir, f"{part}.hdf5"), 'w') as file:

                    for ftype in tqdm(["input_ids", "input_mask", "label_ids"]):
                        counter = 0
                        dtype_dict = {"input_ids": 'i8', "input_mask": 'i1', "label_ids": 'i1'}
                        input_data = f[ftype]
                        output_data = file.create_dataset(ftype, (end-start, 512),
                                                                      maxshape=(1000000, 512),
                                                                      dtype=dtype_dict[ftype])
                        for index in tqdm(idxs[start:end]):
                            a = input_data[index, :]
                            output_data[counter, :] = input_data[index, :]
                            counter += 1



def main():

#    path_to_data = "dataset_plus_correct.txt"
#    data_processor = DataPreprocess(path_to_file=path_to_data)
#    data_processor.process_batch()


    to_train_val_test_hdf(data_dir='./new_data_pow/', output_dir='./new_data_split/', train_part=0.8, val_part=0.2, test_part=0.0, length=140000, random_seed=1)


if __name__ == "__main__":
    main()

