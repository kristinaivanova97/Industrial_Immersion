import os
import re
import random
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from enum import Enum

import h5py
import numpy as np
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Errors(int, Enum):
    error_0 = 0
    error_1 = 1


class TestPreprocess:
    def __init__(self):
        self.label_list = ["[Padding]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya",
                           "REPLACE_tsya",
                           "[##]"]
        self.label_map = {label: i for i, label in enumerate(self.label_list)}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        
    def process(self, text, max_seq_length=512, batch_size=16):
        input_ids_full = []
        attention_masks = []
        # label_ids_full = []
        nopad = []

        # y_label = self.gettags(text)
        for i, sentence in enumerate(text):
            tokens = []
            for j, word in  enumerate(re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                # label_1 = y_label[i][j]
                # for m in range(len(token)):
                #     labels.append(label_1)

            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
            ntokens = []
            label_ids = []
            ntokens.append("[CLS]")
            label_ids.append(self.label_map["[CLS]"])
            for k, token in enumerate(tokens):
                ntokens.append(token)
                    
            ntokens.append("[SEP]")
            nopad.append(len(ntokens))
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                ntokens.append("[Padding]")
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            input_ids_full.append(input_ids)
            attention_masks.append(input_mask)
            # label_ids_full.append(label_ids)
            
        input_ids = torch.tensor(input_ids_full)
        attention_masks = torch.tensor(attention_masks)
        prediction_data = TensorDataset(input_ids, attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        return input_ids, attention_masks, prediction_dataloader, nopad

    def check_contain_tsya_or_nn(self, data):

        data_with_tsya_or_nn = []
        tsya_search = re.compile(r'тся\b')
        tsiya_search = re.compile(r'ться\b')
        nn_search = re.compile(r'\wнн([аоы]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых)\b',
                               re.IGNORECASE)  # the words, which contain "н" in the middle or in the end of word
        n_search = re.compile(r'[аоэеиыуёюя]н([аоы]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых)\b', re.IGNORECASE)

        for sentence in data:

            places_with_tsya = tsya_search.search(sentence)
            places_with_tisya = tsiya_search.search(sentence)
            places_with_n = n_search.search(sentence)
            places_with_nn = nn_search.search(sentence)

            # if any([elem is not None for elem in [places_with_tsya, ]]):
            #     data_with_tsya_or_nn.append(sentence)

            if (places_with_tsya is not None) or (places_with_tisya is not None) or (places_with_n is not None) or (
                    places_with_nn is not None):
                data_with_tsya_or_nn.append(sentence)

        return data_with_tsya_or_nn

class ProcessOutput:

    def __init__(self):

        self._tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    def print_results_in_file(self, file_name, tokens, preds, initial_text, correct_text, message, error):
        print("Tokens = ", tokens, file=file_name)
        print("Prediction = ", preds, file=file_name)
        print("Initial text = {} \n".format(initial_text), file=file_name)
        print("Correct text = {} \n".format(correct_text), file=file_name)
        print(file=file_name)

    def print_results(self, tokens, preds, initial_text, correct_text, message, error):
        print("Answer = ", message)
        print("Tokens = ", tokens)
        print("Prediction = ", preds)
        print("Initial text = {} \n".format(initial_text))
        print("Correct text = {} \n".format(correct_text))
        print("Mistake = {} \n".format(error))

    #def process(self, predictions, input_ids, nopad, data_tags, text_data):
    def process(self, predictions, input_ids, nopad, text_data):
        # with open('results.txt', 'w') as file_name:
        tokens = []
        text = []
        fine_text = ''
        preds = []
        correct_text_full = ''

        incorrect_words_from_sentences = []
        all_messages = []
        all_errors = []

        step = 0

        for i, predict in enumerate(predictions):
            for j, pred in enumerate(predict):
                tokens = self._tokenizer.convert_ids_to_tokens(input_ids[step, :nopad[step]])
                text = self._tokenizer.decode(input_ids[step, :nopad[step]])
                # self.fine_text = self.text.replace('[CLS] ', '').replace(' [SEP]', '')
                initial_text = text_data[step]
                #tags =  np.array(data_tags[step][:nopad[step]])
                preds = np.array(pred)
                correct_text = initial_text
                incorrect_words = []
                incorrect_words_tisya = []
                incorrect_words_tsya = []
                incorrect_words_n = []
                incorrect_words_nn = []
                message = ["Correct"]
                error = []

                replace_tsya = np.where(preds==7)[0].tolist()
                replace_tisya = np.where(preds==6)[0].tolist()
                replace_n = np.where(preds==5)[0].tolist()
                replace_nn = np.where(preds==4)[0].tolist()

                list_of_replace_indeces = [replace_tsya, replace_tisya, replace_n, replace_nn]
                list_of_words_with_mistake = [incorrect_words_tsya, incorrect_words_tisya, incorrect_words_n, incorrect_words_nn]

                for j,replace_list in enumerate(list_of_replace_indeces):

                    if len(replace_list) > 0:
                        message = ["Incorrect"]
                        for i in range(len(replace_list)):
                            word = tokens[replace_list[i]]
                            k = 1
                            #while preds[replace_list[i] + k] == 8: # ["##"]
                            while preds[replace_list[i] + k] == preds[replace_list[i]]:
                                index = replace_list[i] + k
                                word += tokens[index][2:]
                                k+=1
                            if '##' not in word:
                                incorrect_words.append(word)
                                list_of_words_with_mistake[j].append(word)

                for word in incorrect_words_tisya:
                    error.append("Тся -> ться")
                    word_correct = word.replace('тся', 'ться')
                    correct_text = correct_text.replace(word, word_correct)

                for word in incorrect_words_tsya:
                    error.append("Ться -> тся")
                    word_correct = word.replace('ться', 'тся')
                    correct_text = correct_text.replace(word, word_correct)

                pattern_nn = re.compile(r'(?-i:нн)(?=([аоы]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых)\b)', re.IGNORECASE)
                pattern_n = re.compile(r'(?<=[аоэеиыуёюя])(?-i:н)(?=([аоы]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых)\b)', re.IGNORECASE)
                for word in incorrect_words_n:
                    error.append("нн -> н")
                    word_correct = pattern_nn.sub('н', word)
                    #word_correct = word.replace('нн', 'н')
                    correct_text = correct_text.replace(word, word_correct)

                for word in incorrect_words_nn:
                    error.append("н -> нн")
                    word_correct = pattern_n.sub('нн', word)
                    #word_correct = word.replace('н', 'нн')
                    correct_text = correct_text.replace(word, word_correct)

                self.print_results(tokens, preds, initial_text, correct_text, message, error)

                incorrect_words_from_sentences.append(incorrect_words)
                all_messages.append(message)
                all_errors.append(error)
                correct_text_full += correct_text

                step+=1

        return all_messages, incorrect_words_from_sentences, correct_text_full, all_errors
'''
def _check_coincide(self):
    
    coincide = np.sum(self.tags[(self.tags==4) | (self.tags==5)] == self.preds[(self.tags==4) | (self.tags==5)])
    #print("Coincide in {} positions with tsya/tsiya ".format(coincide))
    if coincide == len(self.tags[(self.tags==4) | (self.tags==5)]):
        if (len(self.tags[(self.tags==4) | (self.tags==5)]) == 0):
            print("Sentence does not contain words with tsya/tsiya")
        else:
            print("Predicted and initial sentences coincide")
        return 0
    else:
        print("Sentence contain a mistake!")
        return 1
'''


def permutate(arr, saveOrder=True, seedValue=1):
   idxs = list(range(len(arr)))
   if saveOrder:
      random.seed(seedValue)
   random.shuffle(idxs)
   if isinstance(arr, np.ndarray):
      arr = arr[idxs]
   elif isinstance(arr, list):
      arr = [arr[idx] for idx in idxs]
   else:
      raise TypeError
   return arr

def to_train_val_test_hdf(data_dir = './new_data/', output_dir = './data/', train_part = 0.6,
                          val_part = 0.2, test_part = 0.2, length = 10000, random_seed = 1):


    if not data_dir:
        data_dir = './new_data/'

    if not output_dir:
        output_dir = './data/'

    parts = ["train", "val", "test"]

    with h5py.File(os.path.join(data_dir, f"ids_all.hdf5"), 'r') as f:
        for ftype in tqdm(["input_ids", "input_mask", "label_ids"]):
            dtype_dict = {"input_ids": 'i8', "input_mask": 'i1', "label_ids": 'i1'}
            input_data = f[ftype]

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

                with h5py.File(os.path.join(output_data, f"{part}.hdf5"), 'w') as file:
                    output_data = file.create_dataset(ftype, (length, 512),
                                                                  maxshape=(1000000, 512),
                                                                  dtype=dtype_dict[ftype])
                    output_data[:, :] = input_data[start:end][:, :]

        #
        # with h5py.File(output_dir + 'train' + '.hdf5', 'w') as file_train:
        #     with h5py.File(output_dir + 'val' + '.hdf5', 'w') as file_val:
        #         with h5py.File(output_dir + 'test' + '.hdf5', 'w') as file_test:
        #
        #
        #             for ftype in tqdm(["input_ids", "input_mask", "label_ids"]):
        #                 output_data_train = file_train.create_dataset(ftype, (volume_of_train_data, 512), maxshape=(1000000, 512),
        #                                                               dtype=dtype_dict[ftype])
        #
        #                 output_data_val = file_val.create_dataset(ftype, (volume_of_val_data, 512), maxshape=(25000, 512),
        #                                                           dtype=dtype_dict[ftype])
        #                 output_data_test = file_test.create_dataset(ftype, (volume_of_test_data, 512), maxshape=(25000, 512),
        #                                                             dtype=dtype_dict[ftype])
        #                 input_data = f[ftype]
        #
        #                 idxs = list(range(len(input_data)))
        #
        #                 random.seed(random_seed)
        #                 random.shuffle(idxs)
        #
        #                 counter = 0
        #
        #                 for index in tqdm(idxs[:volume_of_val_data+volume_of_train_data+volume_of_test_data]):
        #                     if counter < volume_of_train_data:
        #                         output_data_train[counter, :] = input_data[index, :]
        #                     elif counter < (volume_of_val_data + volume_of_train_data):
        #                         output_data_val[counter-volume_of_train_data, :] = input_data[index, :]
        #                     elif counter < (volume_of_train_data + volume_of_val_data + volume_of_test_data):
        #                         output_data_test[counter-volume_of_train_data - volume_of_val_data, :] = input_data[index, :]
        #                     counter += 1

