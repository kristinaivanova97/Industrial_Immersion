import os
import random
import json
from pathlib import Path

import h5py
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class DataPreprocess:

    def __init__(self, path_to_file, label_list, tokenizer, num_lines, max_seq_length):

        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.tokenizer = tokenizer
        self.file = path_to_file
        self.num_lines = num_lines
        self.max_seq_length = max_seq_length

    def process_batch(self, output_file, data_dir, part_of_word, file_size=1200000):

        with open(self.file, 'r', encoding='utf-8') as file:
            with h5py.File(data_dir + output_file, 'w') as f:
                print(data_dir + output_file)
                print(file_size)
                dset_input_ids = f.create_dataset("input_ids", (file_size, self.max_seq_length), maxshape=(5500000, self.max_seq_length), dtype='i8')
                dset_input_mask = f.create_dataset("input_mask", (file_size, self.max_seq_length), maxshape=(5500000, self.max_seq_length), dtype='i1')
                dset_label_ids = f.create_dataset("label_ids", (file_size, self.max_seq_length), maxshape=(5500000, self.max_seq_length), dtype='i1')
                line = file.readline()
                stripped_line = line.strip()
                line_list = stripped_line.split()
                i = 0
                pbar = tqdm(total=self.num_lines)
                list_of_words = []
                list_of_labeles = []
                print(dset_input_ids.shape)
                while line:

                    if len(line_list) > 1:
                        list_of_words.append(line_list[0])
                        list_of_labeles.append(line_list[1])
                    else:
                        input_ids, input_mask, label_ids, nopad = self.convert_single_example(sentence=list_of_words,
                                                                                              sentence_labels=list_of_labeles,
                                                                                              part_of_word=part_of_word,
                                                                                              max_seq_length=self.max_seq_length)
                        # КОСТЫЛЬ
                        if i >= file_size - 1:
                            print(i, list_of_labeles, input_ids.shape)
                            dset_input_ids.resize((i + 1, self.max_seq_length))
                            dset_input_mask.resize((i + 1, self.max_seq_length))
                            dset_label_ids.resize((i + 1, self.max_seq_length))

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

    def convert_single_example(self, sentence, sentence_labels, max_seq_length=512, part_of_word=False):

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
        # label_ids.append(self.label_map["O"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            '''
            if labels[i] == "[##]":
                label_ids.append(-100)
            else:
                label_ids.append(self.label_map[labels[i]])
            '''
            label_ids.append(self.label_map[labels[i]])

        ntokens.append("[SEP]")
        nopad.append(len(ntokens))
        label_ids.append(self.label_map["[SEP]"])
        # label_ids.append(self.label_map["O"])
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(0)
            # label_ids.append(self.label_map["O"])
            ntokens.append("[PAD]")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        return np.asarray(input_ids), np.asarray(input_mask), np.asarray(label_ids), np.asarray(nopad)


def to_train_val_test_hdf(data_dir='./new_data/', output_dir='./data/', train_part=0.6,
                          val_part=0.2, length=10000, random_seed=1, use_both_datasets=True, filename="ids_all.hdf5",
                          suffix='', max_seq_length=512):
    parts = ["train", "val", "test"]

    with h5py.File(os.path.join(data_dir, filename), 'r') as f:
        # with h5py.File(os.path.join(data_dir, "ids_all_news.hdf5"), 'r') as f2:

            input_data = f['input_ids']
            idxs = list(range(len(input_data)))
            random.seed(random_seed)
            random.shuffle(idxs)

            if use_both_datasets:
                input_data2 = f2['input_ids']
                idxs2 = list(range(len(input_data2)))
                random.shuffle(idxs2)

            points = (
                int(train_part * length),
                int(train_part * length + val_part * length),
                length
            )

            for params in zip(parts, (0,) + points[:-1], points):

                part, start, end = params
                with h5py.File(os.path.join(output_dir, f"{part}" + suffix + ".hdf5"), 'w') as file:

                    for ftype in tqdm(["input_ids", "input_mask", "label_ids"]):
                        counter = 0
                        dtype_dict = {"input_ids": 'i8', "input_mask": 'i1', "label_ids": 'i1'}
                        output_data = file.create_dataset(ftype, (end - start, self.max_seq_length),
                                                          maxshape=(1000000, self.max_seq_length),
                                                          dtype=dtype_dict[ftype])
                        if use_both_datasets:

                            input_data = f[ftype]
                            input_data2 = f2[ftype]

                            for index in tqdm(idxs[start // 2:end // 2]):
                                # if ftype == 'label_ids':
                                #     buffer = [-100 if x in [0, 1, 2, 8] else x - 3 for x in input_data[index, :]]
                                #     print(buffer)
                                # else:
                                #     buffer = input_data[index, :]
                                # output_data[counter, :] = buffer

                                output_data[counter, :] = input_data[index, :]
                                counter += 1
                            for index in tqdm(idxs2[start // 2:end // 2]):
                                output_data[counter, :] = input_data2[index, :]
                                counter += 1
                        else:

                            input_data = f[ftype]

                            for index in tqdm(idxs[start:end]):
                                output_data[counter, :] = input_data[index, :]
                                counter += 1

def split_maintain_pairs(data_dir='./new_data/', output_dir='./data/', train_part=0.6, val_part=0.2, length=10000,
                         random_seed=1, use_both_datasets=True, filename="ids_all.hdf5", suffix='', max_seq_length=512):
    parts = ["train", "val", "test"]

    with h5py.File(os.path.join(data_dir, filename), 'r') as f:
        with h5py.File(os.path.join(data_dir, "ids_all_news.hdf5"), 'r') as f2:

            input_data = f['input_ids']
            input_incorrect = input_data[:len(input_data)//2]
            input_correct = input_data[len(input_data)//2:]
            print(len(input_incorrect), len(input_correct))
            idxs_incorrect = list(range(len(input_data)//2))
            random.seed(random_seed)
            random.shuffle(idxs_incorrect)
            # idxs_correct = [i + len(input_data) // 2 for i in idxs_incorrect]

            if use_both_datasets:
                input_data2 = f2['input_ids']
                input_incorrect2 = input_data2[:len(input_data2) // 2]
                input_correct2 = input_data2[len(input_data2) // 2:]
                print(len(input_incorrect2), len(input_correct2))
                idxs_incorrect2 = list(range(len(input_data2) // 2))
                random.shuffle(idxs_incorrect2)
                # idxs_correct2 = [i + len(input_data) // 2 for i in idxs_incorrect2]

            points = (
                int(train_part * length),
                int(train_part * length + val_part * length),
                length
            )

            for params in zip(parts, (0,) + points[:-1], points):

                part, start, end = params
                with h5py.File(os.path.join(output_dir, f"{part}" + suffix + ".hdf5"), 'w') as file:

                    for ftype in tqdm(["input_ids", "input_mask", "label_ids"]):
                        counter = 0
                        dtype_dict = {"input_ids": 'i8', "input_mask": 'i1', "label_ids": 'i1'}
                        output_data = file.create_dataset(ftype, (2*(end - start), max_seq_length),
                                                          maxshape=(1000000, max_seq_length),
                                                          dtype=dtype_dict[ftype])
                        if use_both_datasets:

                            input_data = f[ftype]
                            input_incorrect = input_data[:len(input_data) // 2]
                            input_correct = input_data[len(input_data) // 2:]
                            input_data2 = f2[ftype]
                            input_incorrect2 = input_data2[:len(input_data2) // 2]
                            input_correct2 = input_data2[len(input_data2) // 2:]

                            for k, index in enumerate(tqdm(idxs_incorrect[start // 2:end // 2])):

                                output_data[counter, :] = input_incorrect[index, :]
                                counter += 1
                                output_data[counter, :] = input_correct[index, :]
                                counter += 1
                            for index in tqdm(idxs_incorrect2[start // 2:end // 2]):
                                output_data[counter, :] = input_incorrect2[index, :]
                                counter += 1
                                output_data[counter, :] = input_correct2[index, :]
                                counter += 1
                        else:
                            input_data = f[ftype]
                            input_incorrect = input_data[:len(input_data) // 2]
                            input_correct = input_data[len(input_data) // 2:]
                            for index in tqdm(idxs_incorrect[start:end]):
                                output_data[counter, :] = input_incorrect[index, :]
                                counter += 1
                                output_data[counter, :] = input_correct[index, :]
                                counter += 1


def main():
    with open("config_datapreprocess_rus.json") as json_data_file:
        configs = json.load(json_data_file)

    if not configs['from_rubert']:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(**configs['config_of_tokenizer'])

    label_list = configs["label_list"]
    if configs["part_of_word"]:
        label_list.append("[##]")

    Path(configs["full_data_path_hdf"]).mkdir(parents=True, exist_ok=True)
    data_processor = DataPreprocess(path_to_file=configs["path_news"] + configs["file_news"],
                                    label_list=label_list, tokenizer=tokenizer, num_lines=configs["number_of_lines"],
                                    max_seq_length=configs["max_seq_length"])
    data_processor.process_batch(output_file='ids_all_news.hdf5', data_dir=configs["full_data_path_hdf"],
                                 part_of_word=configs["part_of_word"], file_size=configs["news_filesize"])
    print("Finished with news")

    Path(configs["full_data_path_hdf"]).mkdir(parents=True, exist_ok=True)
    data_processor = DataPreprocess(path_to_file=configs["path_magazines"] + configs["file_magazines"],
                                    label_list=label_list, tokenizer=tokenizer, num_lines=configs["number_of_lines"],
                                    max_seq_length=configs["max_seq_length"])
    data_processor.process_batch(output_file='ids_all.hdf5', data_dir=configs["full_data_path_hdf"],
                                 part_of_word=configs["part_of_word"], file_size=configs["magazines_filesize"])
    print("processed")

    # Path(configs["split_data_path_hdf"]).mkdir(parents=True, exist_ok=True)
    # to_train_val_test_hdf(data_dir=configs["full_data_path_hdf"], output_dir=configs["split_data_path_hdf"],
    #                       train_part=configs["train_part"], val_part=configs["val_part"],
    #                       length=configs["length_of_data"], random_seed=1,
    #                       use_both_datasets=configs["use_both_datasets"])

    Path(configs["split_data_path_hdf"]).mkdir(parents=True, exist_ok=True)
    split_maintain_pairs(data_dir=configs["full_data_path_hdf"], output_dir=configs["split_data_path_hdf"],
                         train_part=configs["train_part"], val_part=configs["val_part"],
                         length=configs["length_of_data"], random_seed=1,
                         use_both_datasets=configs["use_both_datasets"],
                         max_seq_length=configs["max_seq_length"])

    print("all in all")


if __name__ == "__main__":
    main()
