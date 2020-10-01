import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class Errors(int, Enum):
#     error_0 = 0
#     error_1 = 1



class TestPreprocess:
    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def process(self, text, max_seq_length=512, batch_size=16):
        input_ids_full = []
        attention_masks = []
        nopad = []
        for i, sentence in enumerate(text):
            tokens = []
            for j, word in enumerate(re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)

            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
            ntokens = ["[CLS]"]
            for k, token in enumerate(tokens[1:]):
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
        # nn_search = re.compile(r'\wнн([аоы]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых)\b', re.IGNORECASE)
        # the words, which contain "н" in the middle or in the end of word
        # n_search = re.compile(r'[аоэеиыуёюя]н([аоы]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых)\b', re.IGNORECASE)
        nn_search = re.compile(r'\wнн([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|'
                               r'ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b',
                               re.IGNORECASE)  # the words, which contain "н" in the middle or in the end of word
        n_search = re.compile(r'[аоэеиыуёюя]н([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|'
                              r'ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b',
                              re.IGNORECASE)

        for sentence in data:

            places_with_tsya = tsya_search.search(sentence)
            places_with_tisya = tsiya_search.search(sentence)
            places_with_n = n_search.search(sentence)
            places_with_nn = nn_search.search(sentence)

            if (places_with_tsya is not None) or (places_with_tisya is not None) or \
                    (places_with_n is not None) or (places_with_nn is not None):
                data_with_tsya_or_nn.append(sentence)

        return data_with_tsya_or_nn


class ProcessOutput:

    def __init__(self, tokenizer, path_to_tsya_vocab, path_to_all_n_nn_words):

        # load dictionaries
        self._tokenizer = tokenizer
        with open(path_to_tsya_vocab, 'r') as f:
            pairs = f.read().splitlines()
        self.tisya_existing_words = set([pair.split('\t')[0] for pair in pairs])
        self.tsya_existing_words = set([pair.split('\t')[1] for pair in pairs])
        with open(path_to_all_n_nn_words, 'r') as f:
            self.n_nn_existing_words = set(f.read().splitlines())

        self.pattern_n_cased = re.compile(
            r'(?<=[аоэеиыуёюя])(?-i:Н)'
            r'(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
            re.IGNORECASE)
        self.pattern_nn_cased = re.compile(
            r'(?-i:НН)'
            r'(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
            re.IGNORECASE)
        self.pattern_nn = re.compile(
            r'(?-i:нн)'
            r'(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
            re.IGNORECASE)
        self.pattern_n = re.compile(
            r'(?<=[аоэеиыуёюя])(?-i:н)'
            r'(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
            re.IGNORECASE)

    def print_results_in_file(self, file_name, tokens, preds, initial_text, correct_text):
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

    def check_contain_mistakes(self, tok_place, probability, probability_o,
                               replace_tsya, replace_tisya, replace_n, replace_nn, threshold,
                               tok_replace_probs, tok_hold_probs):
        tok_error_type = ''
        if (tok_place in replace_tsya) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'ться -> тся'
        elif (tok_place in replace_tisya) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'тся -> ться'
        elif (tok_place in replace_n) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'нн -> н'
        elif (tok_place in replace_nn) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'н -> нн'
        return tok_error_type

    def check_in_dictionary(self, existing_words, word, place, tok_error_type, word_correct, dicty, correct_text,
                            error_prob, hold_prob, default_value):
        if word_correct.lower() in existing_words:
            correct_text = correct_text.replace(word, word_correct)
            dicty[place] = [word + "->" + word_correct, str(error_prob), tok_error_type,
                            len(word_correct.encode("utf8"))]
        elif default_value == 'Incorrect':
            dicty[place] = [word, str(1 - hold_prob), "Ошибка, но исправление невозможно", len(word.encode("utf8"))]
        elif default_value == 'Correct':
            dicty.pop(place)
        return correct_text

    def check_in_dictionary_old(self, existing_words, word, index, tok_error_type, word_correct, dicty,
                                correct_text, error_prob, hold_prob, default_value, incorrect_count, message):

        if word_correct.lower() in existing_words:
            dicty[index] = [word + "->" + word_correct, str(error_prob), tok_error_type,
                            len(word_correct.encode("utf8"))]
            correct_text = correct_text.replace(word, word_correct)
        else:
            if incorrect_count == 1 and default_value == 'Correct':
                message = default_value
            dicty[index] = [word, str(1 - hold_prob), "Ошибка, но исправление невозможно",
                            len(word.encode("utf8"))]

        return message, correct_text

    def get_replace_lists(self, preds):

        replace_tsya = np.where(preds == 7)[0].tolist()
        replace_tisya = np.where(preds == 6)[0].tolist()
        replace_n = np.where(preds == 5)[0].tolist()
        replace_nn = np.where(preds == 4)[0].tolist()
        return replace_tsya, replace_tisya, replace_n, replace_nn

    def process_sentence_optimal(self, prediction, input_ids, nopad, text_data, probabilities, probabilities_o,
                                 default_value, threshold=0.5):

        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0, :nopad[0]])
        correct_text = text_data[0]
        preds = np.array(prediction[0][0])
        replace_tsya, replace_tisya, replace_n, replace_nn = self.get_replace_lists(preds)

        words = []
        words_error = []
        words_probs = []
        incorrect_words = []
        correction_dict = {}
        for tok_place, token in enumerate(tokens):
            tok_replace_probs = []
            tok_hold_probs = []
            word = token
            k = 1
            if '##' not in token:
                tok_error_type = self.check_contain_mistakes(tok_place, probabilities[tok_place],
                                                             probabilities_o[tok_place], replace_tsya,
                                                             replace_tisya, replace_n, replace_nn, threshold,
                                                             tok_replace_probs, tok_hold_probs)

                if tok_place + k < len(tokens):
                    while '##' in tokens[tok_place + k]:
                        word += tokens[tok_place + k][2:]
                        tok_error_type = self.check_contain_mistakes(tok_place, probabilities[tok_place],
                                                                     probabilities_o[tok_place], replace_tsya,
                                                                     replace_tisya, replace_n, replace_nn,
                                                                     threshold, tok_replace_probs, tok_hold_probs)
                        k += 1
                words.append(word)
                if len(tok_error_type) > 0:
                    words_error.append(tok_error_type)
                if len(tok_replace_probs) > 0:

                    correction_dict.setdefault(tok_place, [])
                    word_error_prob = max(tok_replace_probs)
                    words_probs.append(word_error_prob)
                    word_hold_prob = max(tok_hold_probs)
                    incorrect_words.append(word)
                    if tok_error_type == 'ться -> тся':
                        if word in self.tisya_existing_words:
                            word_correct = word.replace('ТЬСЯ', 'ТСЯ').replace('ться', 'тся')
                            correct_text = self.check_in_dictionary(self.tsya_existing_words, word, tok_place,
                                                                    tok_error_type, word_correct,
                                                                    correction_dict, correct_text, word_error_prob,
                                                                    word_hold_prob, default_value)
                        else:
                            correction_dict.pop(tok_place)
                    elif tok_error_type == 'тся -> ться':
                        if word in self.tsya_existing_words:
                            word_correct = word.replace('ТСЯ', 'ТЬСЯ').replace('тся', 'ться')
                            correct_text = self.check_in_dictionary(self.tisya_existing_words, word, tok_place,
                                                                    tok_error_type, word_correct,
                                                                    correction_dict, correct_text, word_error_prob,
                                                                    word_hold_prob, default_value)
                        else:
                            correction_dict.pop(tok_place)

                    elif tok_error_type == 'н -> нн':
                        if word in self.n_nn_existing_words:
                            word_correct = self.pattern_n_cased.sub('НН', word)
                            word_correct = self.pattern_n.sub('нн', word_correct)
                            correct_text = self.check_in_dictionary(self.n_nn_existing_words, word, tok_place,
                                                                    tok_error_type, word_correct,
                                                                    correction_dict, correct_text, word_error_prob,
                                                                    word_hold_prob, default_value)
                        else:
                            correction_dict.pop(tok_place)

                    elif tok_error_type == 'нн -> н':
                        if word in self.n_nn_existing_words:
                            word_correct = self.pattern_nn_cased.sub('Н', word)
                            word_correct = self.pattern_nn.sub('н', word_correct)
                            correct_text = self.check_in_dictionary(self.n_nn_existing_words, word, tok_place,
                                                                    tok_error_type, word_correct,
                                                                    correction_dict, correct_text, word_error_prob,
                                                                    word_hold_prob, default_value)
                        else:
                            correction_dict.pop(tok_place)
        return correct_text, correction_dict, words_error, words_probs

    def process_sentence(self, prediction, input_ids, nopad, text_data, probabilities, probabilities_o,
                         default_value, threshold=0.5):
        # print(probabilities)
        # print(probabilities_o)
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0, :nopad[0]])
        preds = np.array(prediction[0][0])
        correct_text = text_data[0]
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

        replace_tsya, replace_tisya, replace_n, replace_nn = self.get_replace_lists(preds)
        incorrect_count = 0
        probs = []
        probs_o = []

        list_of_replace_indeces = [replace_tsya, replace_tisya, replace_n, replace_nn]
        list_of_words_with_mistake = [incorrect_words_tsya, incorrect_words_tisya,
                                      incorrect_words_n, incorrect_words_nn]

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
                                    # if (replace_list[ids]+k) in replace_list and '#' in tokens[replace_list[ids] + k]:
                                    if (replace_list[ids]+k) in replace_list:
                                        check_contain = True
                                    k += 1

                                if not check_contain:
                                    probs.append(current)
                                    probs_o.append(current_o)
                            else:
                                probs.append(current)
                                probs_o.append(current_o)
                                incorrect_count += 1
                            if '##' not in word:
                                incorrect_words.append(word)
                                list_of_words_with_mistake[p].append(word)

        place = 0
        amount_of_corrections = len(incorrect_words)
        for index, pos in enumerate(places[1:-1]):
            word = words[index + 1]
            if amount_of_corrections > 0:
                if word in incorrect_words_tisya:
                    correction_dict.setdefault(index, [])
                    error.append("Тся -> ться")
                    word_correct = word.replace('ТСЯ', 'ТЬСЯ').replace('тся', 'ться')
                    message, correct_text = self.check_in_dictionary_old(self.tisya_existing_words, word, index,
                                                                         "Тся -> ться", word_correct, correction_dict,
                                                                         correct_text, probs[place], probs_o[place],
                                                                         default_value, incorrect_count, message)
                    place += 1
                    amount_of_corrections -= 1
                elif word in incorrect_words_tsya:
                    correction_dict.setdefault(index, [])
                    error.append("Ться -> тся")
                    word_correct = word.replace('ТЬСЯ', 'ТСЯ').replace('ться', 'тся')
                    message, correct_text = self.check_in_dictionary_old(self.tsya_existing_words, word, index,
                                                                         "Ться -> тся", word_correct, correction_dict,
                                                                         correct_text, probs[place], probs_o[place],
                                                                         default_value, incorrect_count, message)
                    place += 1
                    amount_of_corrections -= 1
                elif word in incorrect_words_n:
                    correction_dict.setdefault(index, [])
                    error.append("нн -> н")
                    word_correct = self.pattern_nn_cased.sub('Н', word)
                    word_correct = self.pattern_nn.sub('н', word_correct)
                    message, correct_text = self.check_in_dictionary_old(self.n_nn_existing_words, word, index,
                                                                         "нн -> н", word_correct, correction_dict,
                                                                         correct_text, probs[place], probs_o[place],
                                                                         default_value, incorrect_count, message)
                    place += 1
                    amount_of_corrections -= 1
                elif word in incorrect_words_nn:
                    correction_dict.setdefault(index, [])
                    error.append("н -> нн")
                    word_correct = self.pattern_n_cased.sub('НН', word)
                    word_correct = self.pattern_n.sub('нн', word_correct)
                    message, correct_text = self.check_in_dictionary_old(self.n_nn_existing_words, word, index,
                                                                         "н -> нн", word_correct, correction_dict,
                                                                         correct_text, probs[place], probs_o[place],
                                                                         default_value, incorrect_count, message)
                    place += 1
                    amount_of_corrections -= 1
        # self.print_results(tokens, preds, initial_text, correct_text, message, error)

        return message, incorrect_words, correct_text, error, probs, probs_o, correction_dict

    def process_batch(self, predictions, input_ids, nopad, text_data, probabilities, probabilities_o):

        correct_text_full = []
        incorrect_words_from_sentences = []
        all_messages = []
        all_errors = []

        step = 0

        for i, predict in enumerate(predictions):
            for j, pred in enumerate(predict):
                probs = []
                probs_o = []
                tokens = self._tokenizer.convert_ids_to_tokens(input_ids[step, :nopad[step]])
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

                replace_tsya, replace_tisya, replace_n, replace_nn = self.get_replace_lists(preds)

                list_of_replace_indeces = [replace_tsya, replace_tisya, replace_n, replace_nn]
                list_of_words_with_mistake = [incorrect_words_tsya, incorrect_words_tisya,
                                              incorrect_words_n, incorrect_words_nn]

                for p, replace_list in enumerate(list_of_replace_indeces):

                    if len(replace_list) > 0:
                        message = ["Incorrect"]
                        for ids in range(len(replace_list)):
                            probs.append(probabilities[ids])
                            probs_o.append(probabilities_o[ids])

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
                                    k += 1
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

                step += 1

        return all_messages, incorrect_words_from_sentences, correct_text_full, all_errors, probs, probs_o

