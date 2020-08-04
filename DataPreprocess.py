import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import h5py
from tqdm import tqdm
from transformers import BertTokenizer

from Class import to_train_val_test_hdf

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

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
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

    # path_train_data = "./dataset.txt"
    # data_processor = DataPreprocess(path_to_file=path_to_data)
    # data_processor.process_batch()

    to_train_val_test_hdf(data_dir='./new_data/', output_dir='./data_2/', volume_of_train_data=100000, volume_of_val_data=20000, volume_of_test_data=20000, random_seed=1)


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
