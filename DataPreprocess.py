import h5py
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
from Class import to_train_val_test_hdf
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class DataPreprocess:
    
    def __init__(self, path_to_file, label_list):

        label_list = label_list
        self.label_map = {label: i for i, label in enumerate(label_list)}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        self.file = path_to_file

    def process_batch(self, output_file, data_dir, part_of_word, file_size=1200000):

        with open(self.file, 'r', encoding='utf-8') as file:
            with h5py.File(data_dir + output_file, 'w') as f:
                dset_input_ids = f.create_dataset("input_ids", (file_size, 512), maxshape=(1500000, 512), dtype='i8')
                dset_input_mask = f.create_dataset("input_mask", (file_size, 512), maxshape=(1500000, 512), dtype='i1')
                dset_label_ids = f.create_dataset("label_ids", (file_size, 512), maxshape=(1500000, 512), dtype='i1')
                line = file.readline()
                stripped_line = line.strip()
                line_list = stripped_line.split()
                i = 0
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
                                                                                              part_of_word=part_of_word)

                        if (i % 10000) == 0:
                            print(i)
                        if i >= file_size-1:
                            print(i, list_of_labeles, input_ids.shape)
                            dset_input_ids.resize((i + 1, 512))
                            dset_input_mask.resize((i + 1, 512))
                            dset_label_ids.resize((i + 1, 512))
                        dset_input_ids[i, :] = input_ids[:]
                        dset_input_mask[i, :] = input_mask[:]
                        dset_label_ids[i, :] = label_ids[:]
                        list_of_words = []
                        list_of_labeles = []
                        i += 1

                    line = file.readline()
                    stripped_line = line.strip()
                    line_list = stripped_line.split()

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
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(0)
            ntokens.append("[PAD]")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        return np.asarray(input_ids), np.asarray(input_mask), np.asarray(label_ids), np.asarray(nopad)


def main():

    with open("config_datapreprocess.json") as json_data_file:
        configs = json.load(json_data_file)
    data_processor = DataPreprocess(path_to_file=configs["path_to_news"], label_list=configs["label_list"])
    data_processor.process_batch(output_file='ids_all_news.hdf5', data_dir=configs["data_path"],
                                 part_of_word=configs["part_of_word"], file_size=configs["news_filesize"])
    print("Finished with news")
    data_processor = DataPreprocess(path_to_file=configs["path_to_magazines"], label_list=configs["label_list"])
    data_processor.process_batch(output_file='ids_all.hdf5', data_dir=configs["data_path"],
                                 part_of_word=configs["part_of_word"], file_size=configs["magazines_filesize"])
    print("processed")
    to_train_val_test_hdf(data_dir=configs["data_path"], output_dir=configs["data_path_split"],
                          train_part=configs["train_part"], val_part=configs["val_part"],
                          length=configs["length_of_data"], random_seed=1,
                          use_both_datasets=configs["use_both_datasets"])
    print("all in all")


if __name__ == "__main__":
    main()

