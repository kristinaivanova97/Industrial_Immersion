import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from transformers import BertTokenizer
from tqdm import tqdm
import h5py
import numpy as np
from testing_class import to_train_val, to_choose_part_of_dataset, to_train_val_test_hdf

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
max_seq_length = 512
data_dir = "./new_data/"
#path_to_data = "./dataset.txt"
path_to_data = "./dataset_plus_correct.txt"
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

                        input_ids, input_mask, label_ids, nopad = self.convert_single_example_with_part_of_word(list_of_words,
                                                                                              list_of_labeles)
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
                    input_ids, input_mask, label_ids, nopad = self.convert_single_example_with_part_of_word(list_of_words, list_of_labeles)
                    input_ids_full.append(input_ids)
                    attention_masks_full.append(input_mask)
                    label_ids_full.append(label_ids)
                    nopad_full.append(nopad)
                    list_of_words = []
                    list_of_labeles = []
        return np.asarray(input_ids_full), np.asarray(attention_masks_full), np.asarray(label_ids_full)


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

    # def save_labels(self, path):
    #
    #     my_file = open(path, 'w', encoding='utf-8')
    #     for raw in tqdm(self.y_label):
    #         for elem in raw:
    #             my_file.write(str(elem))
    #             my_file.write(' ')
    #         my_file.write('\n')
    #     my_file.close()

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


    def save_indices_hdf(self, data_dir):
        input_ids, input_mask, label_ids = self._process() 
        #input_ids = np.array(input_ids)
        #attention_masks = np.array(input_mask)
        #label_ids = np.array(label_ids)
        size = len(input_ids)
        with h5py.File(data_dir + 'ids_all.hdf5', 'w') as f:
                dset_input_ids = f.create_dataset("input_ids", (size, 512), maxshape=(1500000,512), dtype='i8')
                dset_input_mask = f.create_dataset("input_mask", (size, 512), maxshape=(1500000,512), dtype='i1')
                dset_label_ids = f.create_dataset("label_ids", (size, 512), maxshape=(1500000,512), dtype='i1')
                dset_input_ids[:, :] = input_ids[:][:]
                dset_input_mask[:, :] = input_mask[:][:]
                dset_label_ids[:, :] = label_ids[:][:]

        '''
        input_ids, input_mask, label_ids = self._process()    
        input_ids = np.array(input_ids)
        attention_masks = np.array(input_mask)
        label_ids = np.array(label_ids)
        train_size = 150000
        val_size = train_size//8

        with h5py.File(data_dir + 'ids_all_train.hdf5', 'w') as f:
            dset_input_ids = f.create_dataset("input_ids", (train_size, 512), dtype='i8')
            dset_input_ids[:,:] = input_ids[:train_size][:]
            dset_input_mask = f.create_dataset("input_mask", (train_size, 512), dtype='i1')
            dset_input_mask[:,:] = input_mask[:train_size][:]
            dset_label_ids = f.create_dataset("label_ids", (train_size, 512), dtype='i1')
            dset_label_ids[:,:] = label_ids[:train_size][:]
            f.close()
        with h5py.File(data_dir + 'ids_all_val.hdf5', 'w') as f:
            dset_input_ids = f.create_dataset("input_ids", (val_size, 512), dtype='i8')
            dset_input_ids[:,:] = input_ids[train_size:(train_size+val_size)][:]
            dset_input_mask = f.create_dataset("input_mask", (val_size, 512), dtype='i1')
            dset_input_mask[:,:] = input_mask[train_size:(train_size+val_size)][:]
            dset_label_ids = f.create_dataset("label_ids", (val_size, 512), dtype='i1')
            dset_label_ids[:,:] = label_ids[train_size:(train_size+val_size)][:]
            f.close()
        '''

def main():
    
    #path_to_train_data = "./dataset.txt"
    #path_to_train_data = "/mnt/sda/orpho/data/dataset_plus_correct.txt"
    #data_processor = DataPreprocess(path_to_file=path_to_train_data)
    #data_processor.save_indices_hdf(data_dir = data_dir)
    
    
    #data_processor.process_batch()
    to_train_val_test_hdf(data_dir='./new_data/', output_dir='./data_split/', volume_of_train_data=100000, volume_of_val_data=20000, volume_of_test_data=20000, random_seed=1)
    
    # data_processor = DataPreprocess(path_to_file=path_to_data)
    #data_processor.save_indices(ftype='data', data_dir = data_dir)
        
    #to_train_val(data_path='./', output_path='./raw_data/', volume_of_train_data=0.8, volume_of_val_data=0.1, volume_of_test_data=0.1, random_seed=1)
    #to_choose_part_of_dataset(data_path='./raw_data/', output_path='./data/', volume_of_train_data=150000, volume_of_val_data=50000, volume_of_test_data=0)
    #data_processor.save_indices(ftype='data', data_dir = data_dir)
    #data_processor.save_indices(ftype='full_labels', data_dir = data_dir)

if __name__ == "__main__":
    main()

