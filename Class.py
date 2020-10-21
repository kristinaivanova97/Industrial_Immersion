import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import csv
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
                ntokens.append("[PAD]")
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

    def check_contain(self, data):

        data_with_tsya_or_nn = []
        prep_search = re.compile(r'(\ba\b)|(\ban\b)|(\bthe\b)', re.IGNORECASE)
        then_than_search = re.compile(r'(\bthen\b)|(\bthan\b)', re.IGNORECASE)
        bad_bed_search = re.compile(r'(\bbad\b)|(\bbed\b)', re.IGNORECASE)
        live_life_search = re.compile(r'(\blife\b)|(\blive\b)', re.IGNORECASE)
        head_had_search = re.compile(r'(\bhad\b)|(\bhead\b)', re.IGNORECASE)
        career_carrier_search = re.compile(r'(\bcareer\b)|(\bcarrier\b)', re.IGNORECASE)
        # many_much_alotof_search = re.compile(r'(\bmany\b)|(\bmuch\b)|(\blot of\b)', re.IGNORECASE)

        for sentence in data:

            places_with_prep = prep_search.search(sentence)
            places_with_then_than = then_than_search.search(sentence)
            places_with_bad_bed = bad_bed_search.search(sentence)
            places_with_live_life = live_life_search.search(sentence)
            places_with_head_had = head_had_search.search(sentence)
            places_with_career_carrier = career_carrier_search.search(sentence)

            if (places_with_prep is not None) or (places_with_then_than is not None)\
                    or (places_with_bad_bed is not None) or (places_with_live_life is not None)\
                    or (places_with_head_had is not None) or (places_with_career_carrier is not None):
                data_with_tsya_or_nn.append(sentence)

        return data_with_tsya_or_nn


class ProcessOutput:

    def __init__(self, tokenizer):

        self._tokenizer = tokenizer
        self.label_map = {label: i for i, label in enumerate(["[PAD]", "[SEP]", "[CLS]", "O",
                                                              "REPLACE_a_an", "REPLACE_the", "DELETE_prep",
                                                              "INSERT_a_an", "INSERT_the",
                                                              "REPLACE_then", "REPLACE_than",
                                                              "REPLACE_bad", "REPLACE_bed",
                                                              "REPLACE_live", "REPLACE_life",
                                                              "REPLACE_head", "REPLACE_had",
                                                              "REPLACE_career", "REPLACE_carrier"])}
        # self.pattern_n_cased = re.compile(
        #     r'(?<=[аоэеиыуёюя])(?-i:Н)'
        #     r'(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
        #     re.IGNORECASE)
        # self.pattern_nn_cased = re.compile(
        #     r'(?-i:НН)'
        #     r'(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
        #     re.IGNORECASE)

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

    # get word in lower form
    def check_a_an(self, word):
        char = ''
        h_start = re.compile(r'(hacendado)|(haut)|(heir)|(hombre)|(honest)|(honor)|(honour)|'
                             '(horchata)|(hostler)|(hourglass)\w+')
        h_words = ['honnête', 'honni', 'hors', 'hour', 'hourly']
        o_words = ['one', 'once', 'oncer', "one's", 'onefold', 'oneness', 'oner', 'oneself', 'onestage', 'onetime']
        o_start = re.compile(r'\b(one-)|(onerous)\w+')
        y_start = re.compile(r'\b(ytterbium)|(yttrium)\w+')
        y_words = ['yclept', 'ywis']
        u_words = ['uey', 'uighur', 'uigur', 'uinta', 'uintah', 'una', 'unidactyl', 'unidecimal', 'unidentate',
                   'unimanual', 'unimatch', 'unimeter', 'unimucronate', 'uniname', 'uninerved', 'unispiral', 'unisys',
                   'unix']
        u_words_an = ['uitlander', 'unabbreviated', 'unabiding', 'unable', 'unabolished', 'unaesthetic', 'unafraid',
                      'unaggressive', 'unagreeable', 'unaided', 'unalarmed', 'unamortized', 'unanalyzable',
                      'unanchored', 'unanesthetized', 'unanticipated', 'unapplied', 'unapt', 'unarchitectural',
                      'unarchive', 'unarrested', 'unartful', 'unascertained', 'unashamed', 'unasked', 'unaspiring',
                      'unatonable', 'unaudited', 'unaugmented', 'unearned', 'uneatable', 'uneclipsed', 'unedged',
                      'uneducated', 'unemotional', 'unerring', 'unerupted', 'unestimated', 'unetched', 'unethical',
                      'unevaluated', 'unexecuted', 'unexercised', 'unilluminated', 'unirradiated', 'unissued',
                      'unobeyed', 'unoccupied', 'unoften', 'unopposed', 'unoptimized', 'unordered', 'unorganized',
                      'unostentatious', 'unowned', 'ununderstandable', 'ununited', 'unused', 'unyawed']
        u_start = re.compile(r'(unanim)|(uni-)|(unib)|(unic)|(unidensit)|(unidextral)|(unidi)|(unidr)|(unif)|(unig)|'
                             r'(unij)|(unila)|(unile)|(unili)|(unilo)|(unimak)|(unimo)|(uninuclea)|(union)|(unip)|'
                             r'(unique)|(unirational)|(unire)|(uniro)|(unisa)|(unise)|(uniso)|(unist)|(unit)|(univ)\w+')
        u_start_an = re.compile(r'(unaba)|(unabr)|(unabso)|(unacc)|(unachiev)|(unack)|(unacqu)|(unact)|(unad)|(unaff)|'
                                r'(unali)|(unallo)|(unalter)|(unambi)|(unamen)|(unann)|(unanswer)|(unappe)|(unappr)|'
                                r'(unargu)|(unarm)|(unass)|(unatt)|(unauth)|(unavail)|(unave)|(unavoidab)|(unaware)|'
                                r'(unearth)|(uneas)|(uneconomic)|(unedi)|(unela)|(unelect)|(unemb)|(unemploy)|(unen)|'
                                r'(unequa)|(unequi)|(unesc)|(unessential)|(uneven)|(unexa)|(unexc)|(unexhaust)|(unexp)|'
                                r'(unext)|(unidentifi)|(unimagina)|(unimp)|(uninc)|(unind)|(uninf)|(uninh)|(uninitia)|'
                                r'(uninjur)|(uninsp)|(uninst)|(uninsu)|(uninte)|(uninv)|(unobject)|(unobli)|(unobs)|'
                                r'(unobt)|(unobvious)|(unoff)|(unoil)|(unope)|(unori)|(unortho)|(unoxidiz)|(ununif)|'
                                r'(unusab)|(unusual)|(unut)|(unyielding)|(unyoke)|(ux)\w+')
        ucv_words = ['ucayali', 'udon', 'ufa', 'ulan', 'ulan-ude', 'ulianovsk', 'ulyanovsk', 'umami', 'umiak', 'upend',
                     'upender', 'uperize', 'uperizer', 'uperization', 'urartu', 'urartian', 'urumchi', 'usumbura',
                     'uxorial', 'uxoricide', 'uxorious', 'uzbek', 'uzbekistan', 'uzi']
        ucc_words = ['ukraine', 'ukrainian', 'utrecht', 'utricle', 'utricular', 'utriculitis', 'utriculosaccular']
        up_words = ['uc', 'ug', 'uh', 'um', 'up', 'us']
        ux_start = re.compile(r'ux\w+')
        an_digits = ['11', '18']
        an_digit_start = re.compile(r'8\d+')

        if word[0] in ['b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's',
                       't', 'v', 'w', 'x', 'z']:
            char = 'a'
        elif word[0] in ['a', 'e', 'i']:
            char = 'an'
        elif word[0] == 'h':
            if h_start.search(word) is not None:
                char = 'an'
            elif word in h_words:
                char = 'an'
            else:
                char = 'a'
        elif word[0] == 'o':
            if word in o_words:
                char = 'a'
            elif o_start.search(word) is not None:
                char = 'a'
            else:
                char = 'an'
        elif word[0] == 'y':
            if y_start.search(word) is not None:
                char = 'an'
            elif word in y_words:
                char = 'an'
            else:
                char = 'a'
        elif word[0] == 'u':
            if word in u_words:
                char = 'a'
            elif u_start.search(word) is not None:
                char = 'a'
            elif u_start_an.search(word) is not None:
                char = 'an'
            elif word in u_words_an:
                char = 'an'
            elif word[1] not in ['a', 'e', 'i', 'o', 'u']:
                if word in ucv_words:
                    char = 'an'
                elif word[2] in ['a', 'e', 'i', 'o', 'u']:
                    char = 'a'
                elif word[2] == '-':
                    char = 'an'
            elif (word[1] not in ['a', 'e', 'i', 'o', 'u']) and (word[2] not in ['a', 'e', 'i', 'o', 'u']):
                if word in ucc_words:
                    char = 'a'
                else:
                    char = 'an'
            elif word in up_words:
                char = 'an'
            elif ux_start.search(word) is not None:
                char = 'an'
        elif word[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            if (word in an_digits) or (an_digit_start.search(word) is not None):
                char = 'an'
            else:
                char = 'a'
        elif word[1] == '-':
            if word[0] in ['b', 'c', 'd', 'g', 'j', 'k', 'p', 'q', 't', 'u', 'v', 'w', 'y', 'z']:
                char = 'a'
            elif word[0] in ['a', 'e', 'f', 'h', 'i', 'l', 'm', 'n', 'o', 'r', 's', 'x']:
                char = 'an'

        return char

    def check_contain_mistakes(self, tok_place, probability, probability_o,
                               replace_lists, threshold,
                               tok_replace_probs, tok_hold_probs):
        replace_a_an, replace_the, delete_prep, insert_the, insert_a_an, replace_than, replace_then, \
         replace_bad, replace_bed, replace_life, replace_live, replace_head, replace_had, \
         replace_career, replace_carrier = replace_lists
        tok_error_type = ''
        if (tok_place in replace_a_an) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'prep -> a/an'
        elif (tok_place in replace_the) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'prep -> the'
        # elif (tok_place in replace_an) and (probability > threshold):
        #     tok_replace_probs.append(probability)
        #     tok_hold_probs.append(probability_o)
        #     tok_error_type = 'prep -> an'
        elif (tok_place in insert_a_an) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'insert prep a/an'
        # elif (tok_place in insert_an) and (probability > threshold):
        #     tok_replace_probs.append(probability)
        #     tok_hold_probs.append(probability_o)
        #     tok_error_type = 'insert prep an'
        elif (tok_place in insert_the) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'insert prep the'
        elif (tok_place in replace_then) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'than -> then'
        elif (tok_place in replace_than) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'then -> than'
        elif (tok_place in replace_bad) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'bed -> bad'
        elif (tok_place in replace_bed) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'bad -> bed'
        elif (tok_place in replace_head) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'had -> head'
        elif (tok_place in replace_had) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'head -> had'
        elif (tok_place in replace_life) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'live -> life'
        elif (tok_place in replace_live) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'life -> live'
        elif (tok_place in replace_career) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'carrier -> career'
        elif (tok_place in replace_carrier) and (probability > threshold):
            tok_replace_probs.append(probability)
            tok_hold_probs.append(probability_o)
            tok_error_type = 'career -> carrier'
        return tok_error_type

    def check_in_dictionary(self, existing_words, word, place, tok_error_type, word_correct, dicty, correct_text,
                            error_prob, hold_prob, default_value, writer, text_data):
        if word_correct.lower() in existing_words:
            correct_text = correct_text.replace(word, word_correct)
            dicty[place] = [word + "->" + word_correct, str(error_prob), tok_error_type,
                            len(word_correct.encode("utf8"))]
        elif default_value == 'Incorrect':
            # dicty[place] = [word, str(1 - hold_prob), "Ошибка, но исправление невозможно", len(word.encode("utf8"))]
            writer.writerow([word, text_data[0], tok_error_type, error_prob])
            dicty[place] = [word, str(1 - hold_prob), "Nothing to do", tok_error_type]

        elif default_value == 'Correct':
            writer.writerow([word, text_data[0], tok_error_type, error_prob])
            dicty.pop(place)
        return correct_text

    def get_replace_lists(self, preds):
        replace_a_an = np.where(preds == 4)[0].tolist()
        replace_the = np.where(preds == 5)[0].tolist()
        delete_prep = np.where(preds == 6)[0].tolist()
        insert_a_an = np.where(preds == 7)[0].tolist()
# TODO check numbers
        insert_the = np.where(preds == 9)[0].tolist()
        replace_then = np.where(preds == 10)[0].tolist()
        replace_than = np.where(preds == 11)[0].tolist()
        replace_bad = np.where(preds == 12)[0].tolist()
        replace_bed = np.where(preds == 13)[0].tolist()
        replace_live = np.where(preds == 14)[0].tolist()
        replace_life = np.where(preds == 15)[0].tolist()
        replace_head = np.where(preds == 16)[0].tolist()
        replace_had = np.where(preds == 17)[0].tolist()
        replace_career = np.where(preds == 18)[0].tolist()
        replace_carrier = np.where(preds == 19)[0].tolist()

        return (replace_a_an, replace_the, delete_prep, insert_the, insert_a_an, replace_than, replace_then, \
                replace_bad, replace_bed, replace_life, replace_live, replace_head, replace_had, \
                replace_career, replace_carrier)

    def process_sentence_optimal(self, prediction, input_ids, nopad, text_data, probabilities, probabilities_o,
                                 default_value, threshold=0.5, for_stand=False):

        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0, :nopad[0]])
        correct_text = text_data[0]
        preds = np.array(prediction[0][0])
        replace_lists = self.get_replace_lists(preds)
        words = []
        words_error = []
        words_probs = []
        words_hold_probs = []
        incorrect_words = []
        correction_dict = {}
        for tok_place, token in enumerate(tokens):
            tok_replace_probs = []
            tok_hold_probs = []
            word = token
            k = 1
            if '##' not in token:
                tok_error_type = self.check_contain_mistakes(tok_place, probabilities[tok_place],
                                                             probabilities_o[tok_place], replace_lists, threshold,
                                                             tok_replace_probs, tok_hold_probs)

                if tok_place + k < len(tokens):
                    while '##' in tokens[tok_place + k]:
                        word += tokens[tok_place + k][2:]
                        tok_error_type = self.check_contain_mistakes(tok_place, probabilities[tok_place],
                                                                     probabilities_o[tok_place], replace_lists,
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
                    words_hold_probs.append(word_hold_prob)
                    incorrect_words.append(word)
                    with open("words_not_in_dict.csv", 'w', newline='') as csvFile:

                        writer = csv.writer(csvFile)
                        writer.writerow(["word", "sentence", "error", "probability"])
                        if tok_error_type == 'prep -> a/an':
                            # for char in ['AN', 'THE']:
                            char = 'THE'
                            prep = self.check_a_an(tokens[tok_place+1].lower())
                            if prep == 'an':
                                word_correct = word.replace(char, "AN").replace(char.lower(), "an")
                            else:
                                word_correct = word.replace(char, "A").replace(char.lower(), "a")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        # elif tok_error_type == 'prep -> an':
                        #     for char in ['A', 'THE']:
                        #         word_correct = word.replace(char, "AN").replace(char.lower(), "an")
                        #         if word != word_correct:
                        #             break
                        #     correct_text = correct_text.replace(word, word_correct)
                        #     correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                        #                                   tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'prep -> the':
                            for char in ['A', 'AN']:
                                word_correct = word.replace(char, "THE").replace(char.lower(), "the")
                                if word != word_correct:
                                    break
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'insert a/an':
                            prep = self.check_a_an(word.lower())
                            if prep == 'a':
                                if word.isupper():
                                    if tok_place == 0:
                                        word_correct = 'A ' + word
                                    else:
                                        word_correct = ' A ' + word
                                elif word.istitle():
                                    if tok_place == 0:
                                        word_correct = 'A ' + word[0].lower() + word[1:]
                                    else:
                                        word_correct = ' A ' + word[0].lower() + word[1:]
                                else:
                                    if tok_place == 0:
                                        word_correct = 'a ' + word
                                    else:
                                        word_correct = ' a ' + word
                            else:
                                if word.isupper():
                                    if tok_place == 0:
                                        word_correct = 'AN ' + word
                                    else:
                                        word_correct = ' AN ' + word
                                elif word.istitle():
                                    if tok_place == 0:
                                        word_correct = 'AN ' + word[0].lower() + word[1:]
                                    else:
                                        word_correct = ' AN ' + word[0].lower() + word[1:]
                                else:
                                    if tok_place == 0:
                                        word_correct = 'an ' + word
                                    else:
                                        word_correct = ' an ' + word
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'insert the':
                            if word.isupper():
                                if tok_place == 0:
                                    word_correct = 'THE ' + word
                                else:
                                    word_correct = ' THE ' + word
                            elif word.istitle():
                                if tok_place == 0:
                                    word_correct = 'The ' + word[0].lower() + word[1:]
                                else:
                                    word_correct = ' the ' + word
                            else:
                                if tok_place == 0:
                                    word_correct = 'the ' + word
                                else:
                                    word_correct = ' the ' + word
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'delete_prep':
                            correct_text = correct_text.replace(word+' ', '')
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'then -> than':
                            word_correct = word.replace('THEN', "THAN").replace('Then', 'Than').replace('then', "than")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'than -> then':
                            word_correct = word.replace('THAN', "THEN").replace('Than', 'Then').replace('than', "then")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'bad -> bed':
                            word_correct = word.replace('BAD', "BED").replace('Bad', 'Bed').replace('bad', "bed")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'bed -> bad':
                            word_correct = word.replace('BED', "BAD").replace('Bed', 'Bad').replace('bed', "bad")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'live -> life':
                            word_correct = word.replace('LIVE', "LIFE").replace('Live', 'Life').replace('live', "life")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'life -> live':
                            word_correct = word.replace('LIFE', "LIVE").replace('Life', 'Live').replace('life', "live")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'head -> had':
                            word_correct = word.replace('HEAD', "HAD").replace('Head', 'Had').replace('head', "had")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'had -> head':
                            word_correct = word.replace('HAD', "HEAD").replace('Had', 'Head').replace('had', "head")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'career -> carrier':
                            word_correct = word.replace('CAREER', "CARRIER").replace('Career', 'Carrier').\
                                replace('career', "carrier")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
                        elif tok_error_type == 'carrier -> career':
                            word_correct = word.replace('CARRIER', "CAREER").replace('Carrier', 'Career').\
                                replace('carrier', "career")
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, len(word_correct.encode("utf8"))]
        return correct_text, correction_dict, words_error, words_probs, words_hold_probs

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
                            while '##' in tokens[replace_list[ids] - k]:
                                word = tokens[replace_list[ids] - k] + word[2:]
                                if (replace_list[ids] - k) in replace_list:
                                    if current < probabilities[replace_list[ids] - k]:
                                        current = probabilities[replace_list[ids] - k]
                                    if current_o < probabilities_o[replace_list[ids] - k]:
                                        current_o = probabilities_o[replace_list[ids] - k]
                                k += 1
                            word = tokens[replace_list[ids] - k] + word[2:]
                        k = 1
                        if replace_list[ids] + k < len(tokens):
                            if '##' in tokens[replace_list[ids] + k]:
                                check_contain = False
                                while '##' in tokens[replace_list[ids] + k]:
                                    word += tokens[replace_list[ids] + k][2:]
                                    # if (replace_list[ids]+k) in replace_list and '#' in tokens[replace_list[ids] + k]:
                                    if (replace_list[ids] + k) in replace_list:
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
                                while '##' in tokens[replace_list[ids] - k]:
                                    word = tokens[replace_list[ids] - k] + word[2:]
                                    k += 1
                                word = tokens[replace_list[ids] - k] + word[2:]
                            k = 1
                            if replace_list[ids] + k < len(tokens):
                                while '##' in tokens[replace_list[ids] + k]:
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
