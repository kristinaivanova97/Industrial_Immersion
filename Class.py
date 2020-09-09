import os
import re
import random
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from enum import Enum

import h5py
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Errors(int, Enum):
    error_0 = 0
    error_1 = 1


class TestPreprocess:
    def __init__(self, label_list):
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(self.label_list)}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        
    def process(self, text, max_seq_length=512, batch_size=16):
        input_ids_full = []
        attention_masks = []
        nopad = []
        for i, sentence in enumerate(text):
            tokens = []
            for j, word in  enumerate(re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)

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
        # nn_search = re.compile(r'\wнн([аоы]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых)\b', re.IGNORECASE) # the words, which contain "н" in the middle or in the end of word
        # n_search = re.compile(r'[аоэеиыуёюя]н([аоы]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых)\b', re.IGNORECASE)
        nn_search = re.compile(r'\wнн([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b', re.IGNORECASE) # the words, which contain "н" in the middle or in the end of word
        n_search = re.compile(r'[аоэеиыуёюя]н([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b', re.IGNORECASE)

        for sentence in data:

            places_with_tsya = tsya_search.search(sentence)
            places_with_tisya = tsiya_search.search(sentence)
            places_with_n = n_search.search(sentence)
            places_with_nn = nn_search.search(sentence)

            if (places_with_tsya is not None) or (places_with_tisya is not None) or (places_with_n is not None) or (places_with_nn is not None):
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
        if len(error) == 0:
            error = ['None']
        print("Mistake = {} \n".format(error))

    def process_sentence(self, prediction, input_ids, nopad, text_data, probabilities, probabilities_o,
                         default_value, threshold=0.5):
        # print(probabilities)
        # print(probabilities_o)

        # load dictionaries
        with open('data/tsya_vocab.txt', 'r') as f:
            pairs = f.read().splitlines()
        tisya_existing_words = set([pair.split('\t')[0] for pair in pairs])
        tsya_existing_words = set([pair.split('\t')[1] for pair in pairs])
        with open('data/all_n_nn_words_full_endings.txt', 'r') as f:
            n_nn_existing_words = set(f.read().splitlines())

        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0, :nopad[0]])
        initial_text = text_data[0]
        preds = np.array(prediction[0][0])
        correct_text = initial_text
        incorrect_words = []
        incorrect_words_tisya = []
        incorrect_words_tsya = []
        incorrect_words_n = []
        incorrect_words_nn = []
        message = "Correct"
        error = []
        correction_dict = {}
        places = []
        words = []
        for pos, token in enumerate(tokens):
            word = token
            k = 1
            if '##' not in token:
                places.append(pos)
            if pos + k < len(tokens):
                while '##' in tokens[pos + k]:
                    index = pos + k
                    word += tokens[index][2:]
                    k += 1
            if '##' not in word:
                words.append(word)

        replace_tsya = np.where(preds == 7)[0].tolist()
        replace_tisya = np.where(preds == 6)[0].tolist()
        replace_n = np.where(preds == 5)[0].tolist()
        replace_nn = np.where(preds == 4)[0].tolist()
        incorrect_count = len(replace_tsya) + len(replace_tisya) + len(replace_n) + len(replace_nn)
        # replace_tsya = np.where(preds==4)[0].tolist()
        # replace_tisya = np.where(preds==3)[0].tolist()
        # replace_n = np.where(preds==2)[0].tolist()
        # replace_nn = np.where(preds==1)[0].tolist()
        probs = []
        probs_o = []

        list_of_replace_indeces = [replace_tsya, replace_tisya, replace_n, replace_nn]
        list_of_words_with_mistake = [incorrect_words_tsya, incorrect_words_tisya, incorrect_words_n, incorrect_words_nn]

        for p, replace_list in enumerate(list_of_replace_indeces):

            if len(replace_list) > 0:
                message = "Incorrect"
                for ids in range(len(replace_list)):
                    if probabilities[replace_list[ids]] > threshold:
                        word = tokens[replace_list[ids]]
                        k = 1
                        current = probabilities[replace_list[ids]]
                        current_o = probabilities_o[replace_list[ids]]
                        if '##' in word:
                            while '##' in tokens[replace_list[ids]-k]:
                                word = tokens[replace_list[ids]-k]+word[2:]
                                if (replace_list[ids]-k) in replace_list:
                                    if current < probabilities[replace_list[ids]-k]:
                                        current = probabilities[replace_list[ids]-k]
                                    if current_o < probabilities_o[replace_list[ids] - k]:
                                        current_o = probabilities_o[replace_list[ids] - k]
                                k += 1
                            word = tokens[replace_list[ids]-k] + word[2:]
                        k = 1
                        if replace_list[ids] + k < len(tokens):
                            if '##' in tokens[replace_list[ids] + k]:
                                check_contain = False
                                while '##' in tokens[replace_list[ids] + k]:
                                    word += tokens[replace_list[ids] + k][2:]
                                    k += 1
                                    if (replace_list[ids]+k) in replace_list:
                                        check_contain = True
                                if not check_contain:
                                    probs.append(current)
                                    probs_o.append(current_o)
                            else:
                                probs.append(current)
                                probs_o.append(current_o)
                        if '##' not in word:
                            incorrect_words.append(word)
                            list_of_words_with_mistake[p].append(word)

        pattern_n_cased = re.compile(
            r'(?<=[аоэеиыуёюя])(?-i:Н)(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
            re.IGNORECASE)
        pattern_nn_cased = re.compile(
            r'(?-i:НН)(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
            re.IGNORECASE)
        pattern_nn = re.compile(
            r'(?-i:нн)(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
            re.IGNORECASE)
        pattern_n = re.compile(
            r'(?<=[аоэеиыуёюя])(?-i:н)(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
            re.IGNORECASE)
        place = 0

        for index, pos in enumerate(places[1:-1]):
            word = words[index + 1]
            if word in incorrect_words_tisya:
                correction_dict.setdefault(index, [])
                error.append("Тся -> ться")
                word_correct = word.replace('ТСЯ', 'ТЬСЯ').replace('тся', 'ться')
                if word_correct.lower() in tisya_existing_words:
                    correct_text = correct_text.replace(word, word_correct)
                    correction_dict[index] = [word + "->" + word_correct, str(probs[place]), "Тся -> ться",
                                              len(word_correct.encode("utf8"))]
                else:
                    if incorrect_count == 1 and default_value == 'Correct':
                        message = default_value
                    correction_dict[index] = [word, str(1 - probs_o[place]), "Ошибка, но исправление невозможно",
                                              len(word.encode("utf8"))]
                place += 1
            elif word in incorrect_words_tsya:
                correction_dict.setdefault(index, [])
                error.append("Ться -> тся")
                word_correct = word.replace('ТЬСЯ', 'ТСЯ').replace('ться', 'тся')
                if word_correct.lower() in tsya_existing_words:
                    correction_dict[index] = [word + "->" + word_correct, str(probs[place]), "Ться -> тся",
                                              len(word_correct.encode("utf8"))]
                    correct_text = correct_text.replace(word, word_correct)
                else:
                    if incorrect_count == 1 and default_value == 'Correct':
                        message = default_value
                    correction_dict[index] = [word, str(1 - probs_o[place]), "Ошибка, но исправление невозможно",
                                              len(word.encode("utf8"))]
                place += 1
            elif word in incorrect_words_n:
                correction_dict.setdefault(index, [])
                error.append("нн -> н")
                word_correct = pattern_nn_cased.sub('Н', word)
                word_correct = pattern_nn.sub('н', word_correct)
                if word_correct.lower() in n_nn_existing_words:
                    correction_dict[index] = [word + "->" + word_correct, str(probs[place]), "нн -> н",
                                              len(word_correct.encode("utf8"))]
                    correct_text = correct_text.replace(word, word_correct)
                else:
                    if incorrect_count == 1 and default_value == 'Correct':
                        message = default_value
                    correction_dict[index] = [word, str(1 - probs_o[place]), "Ошибка, но исправление невозможно",
                                              len(word.encode("utf8"))]
                place += 1
            elif word in incorrect_words_nn:
                correction_dict.setdefault(index, [])
                error.append("н -> нн")
                word_correct = pattern_n_cased.sub('НН', word)
                word_correct = pattern_n.sub('нн', word_correct)
                if word_correct.lower() in n_nn_existing_words:
                    correction_dict[index] = [word + "->" + word_correct, str(probs[place]), "н -> нн",
                                              len(word_correct.encode("utf8"))]
                    correct_text = correct_text.replace(word, word_correct)
                else:
                    if (incorrect_count == 1) and (default_value == 'Correct'):
                        message = default_value
                    correction_dict[index] = [word, str((1 - probs_o[place])), "Ошибка, но исправление невозможно",
                                              len(word.encode("utf8"))]
                place += 1
        #self.print_results(tokens, preds, initial_text, correct_text, message, error)

        return message, incorrect_words, correct_text, error, probs, probs_o, correction_dict

    def process_batch(self, predictions, input_ids, nopad, text_data, probabilities, probabilities_O):

        correct_text_full = []
        incorrect_words_from_sentences = []
        all_messages = []
        all_errors = []

        step = 0

        for i, predict in enumerate(predictions):
            for j, pred in enumerate(predict):
                probs = []
                probs_O = []
                tokens = self._tokenizer.convert_ids_to_tokens(input_ids[step, :nopad[step]])
                # text = self._tokenizer.decode(input_ids[step, :nopad[step]])
                # self.fine_text = self.text.replace('[CLS] ', '').replace(' [SEP]', '')
                initial_text = text_data[step]
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
                # replace_tsya = np.where(preds==4)[0].tolist()
                # replace_tisya = np.where(preds==3)[0].tolist()
                # replace_n = np.where(preds==2)[0].tolist()
                # replace_nn = np.where(preds==1)[0].tolist()

                list_of_replace_indeces = [replace_tsya, replace_tisya, replace_n, replace_nn]
                list_of_words_with_mistake = [incorrect_words_tsya, incorrect_words_tisya, incorrect_words_n, incorrect_words_nn]

                for p, replace_list in enumerate(list_of_replace_indeces):

                    if len(replace_list) > 0:
                        message = ["Incorrect"]
                        for ids in range(len(replace_list)):
                            probs.append(probabilities[ids])
                            probs_O.append(probabilities_O[ids])

                            word = tokens[replace_list[ids]]
                            k = 1
                            if '##' in word:
                                while '##' in tokens[replace_list[ids]-k]:
                                    word = tokens[replace_list[ids]-k]+word[2:]
                                    k += 1
                                word = tokens[replace_list[ids]-k] + word[2:]
                            k = 1
                            if replace_list[ids]+k < len(tokens):
                                while '##' in tokens[replace_list[ids]+k]:
                                    index = replace_list[ids] + k
                                    word += tokens[index][2:]
                                    k+=1
                            if '##' not in word:
                                incorrect_words.append(word)
                                list_of_words_with_mistake[p].append(word)

                for word in incorrect_words_tisya:
                    error.append("Тся -> ться")
                    word_correct = word.replace('ТСЯ', 'ТЬСЯ').replace('тся', 'ться')
                    correct_text = correct_text.replace(word, word_correct)

                for word in incorrect_words_tsya:
                    error.append("Ться -> тся")
                    word_correct = word.replace('ТЬСЯ', 'ТСЯ').replace('ться', 'тся')
                    correct_text = correct_text.replace(word, word_correct)

                pattern_n_cased = re.compile(r'(?<=[аоэеиыуёюя])(?-i:Н)(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
                                           re.IGNORECASE)
                pattern_nn_cased = re.compile(r'(?-i:НН)(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)', re.IGNORECASE)
                pattern_nn = re.compile(r'(?-i:нн)(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)', re.IGNORECASE)
                pattern_n = re.compile(r'(?<=[аоэеиыуёюя])(?-i:н)(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)', re.IGNORECASE)
                for word in incorrect_words_n:
                    error.append("нн -> н")
                    word_correct = pattern_nn_cased.sub('Н', word)
                    word_correct = pattern_nn.sub('н', word_correct)
                    correct_text = correct_text.replace(word, word_correct)

                for word in incorrect_words_nn:
                    error.append("н -> нн")
                    word_correct = pattern_n_cased.sub('НН', word)
                    word_correct = pattern_n.sub('нн', word_correct)
                    correct_text = correct_text.replace(word, word_correct)

                self.print_results(tokens, preds, initial_text, correct_text, message, error)

                incorrect_words_from_sentences.append(incorrect_words)
                all_messages.append(message)
                all_errors.append(error)
                correct_text_full.append(correct_text)

                step+=1

        return all_messages, incorrect_words_from_sentences, correct_text_full, all_errors,probs, probs_O

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
                          val_part=0.2, length=10000, random_seed=1, use_both_datasets=True):


    if not data_dir:
        data_dir = './new_data/'

    if not output_dir:
        output_dir = './data/'

    parts = ["train", "val", "test"]

    with h5py.File(os.path.join(data_dir, "ids_all.hdf5"), 'r') as f:
        with h5py.File(os.path.join(data_dir, "ids_all_news.hdf5"), 'r') as f2:

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
                with h5py.File(os.path.join(output_dir, f"{part}.hdf5"), 'w') as file:

                    for ftype in tqdm(["input_ids", "input_mask", "label_ids"]):
                        counter = 0
                        dtype_dict = {"input_ids": 'i8', "input_mask": 'i1', "label_ids": 'i1'}
                        output_data = file.create_dataset(ftype, (end-start, 512),
                                                                      maxshape=(1000000, 512),
                                                                      dtype=dtype_dict[ftype])
                        if use_both_datasets:

                            input_data = f[ftype]
                            input_data2 = f2[ftype]

                            for index in tqdm(idxs[start//2:end//2]):

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
