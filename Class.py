import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import re
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import random

weight_path = "Chkpt.pth"
batch_size = 16
max_seq_length = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestPreprocess:
    
    def __init__(self):

        self.label_list = ["[Padding]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya",
                           "REPLACE_tsya",
                           "[##]"]
        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        
    def process(self, text, max_seq_length = max_seq_length, batch_size = batch_size):

        input_ids_full = []
        attention_masks = []
        # label_ids_full = []
        nopad = []

        # y_label = self.gettags(text)
        for i, sentence in enumerate(text):
            tokens = []
            labels = []
            for j, word in  enumerate(re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                # label_1 = y_label[i][j]
                # for m in range(len(token)):
                #     labels.append(label_1)

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
            input_ids_full.append(input_ids)
            attention_masks.append(input_mask)
            # label_ids_full.append(label_ids)
            
        input_ids = torch.tensor(input_ids_full)
        attention_masks = torch.tensor(attention_masks)
        prediction_data = TensorDataset(input_ids, attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        return input_ids, attention_masks, prediction_dataloader, nopad


class ProcessOutput:

    def __init__(self):

        self._tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    def print_results_in_file(self, file_name):
        print("Tokens = ", tokens, file=file_name)
        print("Prediction = ", preds, file=file_name)
        print("Initial text = {} \n".format(initial_text), file=file_name)
        print("Correct text = {} \n".format(correct_text), file=file_name)
        print(file=file_name)

    def print_results(self, tokens, preds, initial_text, correct_text, message):
        print("Answer = ", message)
        print("Tokens = ", tokens)
        print("Prediction = ", preds)
        print("Initial text = {} \n".format(initial_text))
        print("Correct text = {} \n".format(correct_text))

    #def process(self, predictions, input_ids, nopad, data_tags, text_data):
    def process(self, predictions, input_ids, nopad, text_data):
        # with open('results.txt', 'w') as file_name:
        tokens = []
        text = []
        fine_text = ''
        preds = []
        correct_text_full = ''
       
        incorrect_for_sentences = []
        all_messages = []
        all_errors = []
        
        step = 0
        
        for i,predict in enumerate(predictions):
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
                            while preds[replace_list[i] + k] == 8: # ["##"]
                                index = replace_list[i] + k
                                word += tokens[index][2:]
                                k+=1
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

                self.print_results(tokens, preds, initial_text, correct_text, message)
                
                incorrect_for_sentences.append(incorrect_words)
                all_messages.append(message)
                all_errors.append(error)
                correct_text_full += correct_text
                
                step+=1

        return incorrect_for_sentences, all_messages, correct_text_full, all_errors
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

def to_train_val(data_path, output_path, volume_of_train_data, volume_of_val_data, volume_of_test_data, random_seed = 1):

    if not data_path:
        data_path = './'
    if not output_path:
        output_path = './raw_data/'

    file_names = ['input_ids_', 'input_mask_',
                  'label_ids_']

    for file_name in file_names:

        input_file_lines = open(data_path + file_name + 'data.txt', 'r', encoding='utf-8').readlines()

        permutated_input_file_lines = permutate(input_file_lines, saveOrder=True, seedValue=random_seed)

        output_file_train_lines = open(output_path + file_name + 'train' + '.txt', 'w', encoding='utf-8')
        output_file_val_lines = open(output_path + file_name + 'val' + '.txt', 'w', encoding='utf-8')
        output_file_test_lines = open(output_path + file_name + 'test' + '.txt', 'w', encoding='utf-8')

        counter = 0
        for line in permutated_input_file_lines:
            if counter < volume_of_train_data*len(permutated_input_file_lines):
                output_file_train_lines.writelines(line)
            elif counter < (volume_of_val_data + volume_of_train_data)*len(permutated_input_file_lines):
                output_file_val_lines.writelines(line)
            elif counter < (volume_of_train_data + volume_of_val_data + volume_of_test_data)*len(permutated_input_file_lines):
                output_file_test_lines.writelines(line)
            counter += 1
        output_file_train_lines.close()
        output_file_val_lines.close()
        output_file_val_lines.close()

def to_choose_part_of_dataset(data_path, output_path, volume_of_train_data, volume_of_val_data, volume_of_test_data):

    dict_config = {'train': volume_of_train_data, 'val': volume_of_val_data, 'test':volume_of_test_data}

    if not data_path:
        data_path = './raw_data/'
    if not output_path:
        output_path = './data/'

    file_names = ['input_ids_', 'input_mask_',
                  'label_ids_']

    for ftype in ['train', 'val', 'test']:
        for file_name in file_names:

            input_file_lines = open(data_path + file_name + ftype +'.txt', 'r', encoding='utf-8').readlines()

            output_file_lines = open(output_path + file_name + ftype + '.txt', 'w', encoding='utf-8')
            counter = 0
            for line in input_file_lines:
                if counter < dict_config[ftype]:
                    output_file_lines.writelines(line)
                else:
                    break
                counter += 1
