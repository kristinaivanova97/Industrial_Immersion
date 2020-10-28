import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import csv
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    wordnet
except Exception:
   print("nltk.download('wordnet')", "nltk.download('averaged_perceptron_tagger')")


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
        many_much_alotof_search = re.compile(r'(\bmany\b)|(\bmuch\b)|(\blot\s+of\b)', re.IGNORECASE)
        inonatofforto_search = re.compile(r'(\bin\b)|(\bon\b)|(\bat\b)|(\bof\b)|(\bfor\b)|(\bTO\b)', re.IGNORECASE)

        for sentence in data:

            places_with_prep = prep_search.search(sentence)
            places_with_then_than = then_than_search.search(sentence)
            places_with_bad_bed = bad_bed_search.search(sentence)
            places_with_live_life = live_life_search.search(sentence)
            places_with_head_had = head_had_search.search(sentence)
            places_with_career_carrier = career_carrier_search.search(sentence)
            places_with_many_much_alotof = many_much_alotof_search.search(sentence)
            places_with_inonatofforto = inonatofforto_search.search(sentence)

            if (places_with_prep is not None) or (places_with_then_than is not None)\
                    or (places_with_bad_bed is not None) or (places_with_live_life is not None)\
                    or (places_with_head_had is not None) or (places_with_career_carrier is not None)\
                    or (places_with_many_much_alotof is not None) or (places_with_inonatofforto is not None):
                data_with_tsya_or_nn.append(sentence)

        return data_with_tsya_or_nn


class ProcessOutput:

    def __init__(self, tokenizer):

        self._tokenizer = tokenizer
        with open("mistake_tags.json") as json_data_file:
            configs = json.load(json_data_file)
        self.label_list = configs["label_list"]

        self.label_map = {label: i for i, label in enumerate(self.label_list)}
        self.error_types = {i+3: error for i, error in enumerate(configs["error_types"])}

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

    def check_contain_mistakes(self, tok_place, probability, probability_o, replace_lists, threshold,
                               tok_replace_probs, tok_hold_probs):
        tok_error_type = ''
        for key in replace_lists:
            if (tok_place in replace_lists[key]) and (probability > threshold):
                tok_replace_probs.append(probability)
                tok_hold_probs.append(probability_o)
                tok_error_type = self.error_types[str(key)]
        return tok_error_type

    def get_replace_lists(self, preds):
        mistakes = {i: np.where(preds == i)[0].tolist() for i in range(4, len(self.label_list))}
        return mistakes

    def upper_or_lower(self, word, tok_place, inserted):

        if word.isupper():
            if tok_place == 0:
                corrected = inserted.upper()+' ' + word
            else:
                corrected = ' '+inserted.upper()+' ' + word
        elif word.istitle():
            if tok_place == 0:
                corrected = inserted[0].upper()+inserted[1:]+' ' + word[0].lower() + word[1:]
            else:
                corrected = ' '+inserted+' ' + word
        else:
            if tok_place == 0:
                corrected = inserted+' ' + word
            else:
                corrected = ' '+inserted+' ' + word
        return corrected

    def replace_multiple(self, tobereplaced, replace_char, word, dicty, place, word_error_prob, correct_text,
                         tok_error_type, positional_symbols):
        word_correct = ''
        for char in tobereplaced:
            word_correct = word.replace(char, replace_char.upper()).\
                replace(char[0] + char[1:].lower(), replace_char[0].upper()+replace_char[1:]). \
                replace(char.lower(), replace_char)
            if word != word_correct:
                break
        correct_text = correct_text.replace(word, word_correct)
        start = positional_symbols
        dicty[place] = [word + "->" + word_correct, str(word_error_prob), tok_error_type, start, 
                        len(word.encode("utf8"))]
        return dicty

    def multiple_insert(self, insert_list, tok_error_type, inserted_char_list, word, place, word_error_prob, dicty,
                        correct_text, positional_symbols):
        inserted_ids = insert_list.index(tok_error_type)
        inserted_char = inserted_char_list[inserted_ids]
        word_correct = self.upper_or_lower(word, place, inserted_char)
        correct_text = correct_text.replace(word, word_correct)
        start = positional_symbols
        dicty[place] = [word + "->" + word_correct, str(word_error_prob), tok_error_type, start,
                        len(word.encode("utf8"))]
        return dicty

    def choose_comparative_form(self, word, compare_type):  # word in lower form
        vowel = re.compile("[aeiouy]", re.IGNORECASE)
        answer = vowel.search(word)
        word_correct = ''
        if answer is not None:
            double_check = word.replace(answer.group(), '')
            if vowel.search(double_check[:-1]) is not None:  # двухсложные прилагательные и наречия
                if word[-1] == 'y':
                    if compare_type == 'most' or compare_type == 'least':
                        word_correct = word[:-1] + 'iest'
                    elif compare_type == 'more' or compare_type == 'less':
                        word_correct = word[:-1] + 'ier'
                elif word[-1] == 'e':
                    if compare_type == 'most' or compare_type == 'least':
                        word_correct = word[:-1] + 'est'
                    elif compare_type == 'more' or compare_type == 'less':
                        word_correct = word[:-1] + 'er'
                else:
                    if compare_type == 'most' or compare_type == 'least':
                        word_correct = word + 'est'
                    elif compare_type == 'more' or compare_type == 'less':
                        word_correct = word + 'er'
            else:
                if word[-1] == 'y':
                    if compare_type == 'most' or compare_type == 'least':
                        word_correct = word[:-1] + 'iest'
                    elif compare_type == 'more' or compare_type == 'less':
                        word_correct = word[:-1] + 'ier'
                elif word[-1] == 'e':
                    if compare_type == 'most' or compare_type == 'least':
                        word_correct = word[:-1] + 'est'
                    elif compare_type == 'more' or compare_type == 'less':
                        word_correct = word[:-1] + 'er'
                elif (word[-1] not in ['a', 'e', 'i', 'o', 'u']) and (word[-1] in ['a', 'e', 'i', 'o', 'u', 'y']):
                    if compare_type == 'most' or compare_type == 'least':
                        word_correct = word + word[-1] + 'est'
                    elif compare_type == 'more' or compare_type == 'less':
                        word_correct = word + word[-1] + 'er'
                else:
                    if compare_type == 'most' or compare_type == 'least':
                        word_correct = word + 'est'
                    elif compare_type == 'more' or compare_type == 'less':
                        word_correct = word + 'er'
        else:
            if compare_type == 'most' or compare_type == 'least':
                word_correct = word + 'est'
            elif compare_type == 'more' or compare_type == 'less':
                word_correct = word + 'er'
        return word_correct

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

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
        corrected_words = []
        correction_dict = {}
        pron_insert_list = ["insert_pron it", "insert_pron he", "insert_pron she", "insert_pron I",
                            "insert_pron we", "insert_pron you", "insert_pron they", "insert_pron its",
                            "insert_pron his", "insert_pron her", "insert_pron my", "insert_pron our",
                            "insert_pron your", "insert_pron their", "insert_pron him", "insert_pron me",
                            "insert_pron us", "insert_pron them", "insert_pron this",
                            "insert_pron these", "insert_pron that", "insert_pron those",
                            "insert_pron who", "insert_pron whom", "insert_pron which",
                            "insert_pron ones", "insert_pron one"]
        prep_insert_list = ["insert_prep in", "insert_prep on", "insert_prep at", "insert_prep of", "insert_prep to",
                            "insert_prep for", "insert_prep from", "insert_prep by", "insert_prep with",
                            "insert_prep about", "insert_prep off", "insert_prep down", "insert_prep up",
                            "insert_prep upon", "insert_prep within", "insert_prep above", "insert_prep below"]
        verb_insert_list = ["insert_verb is", "insert_verb are", "insert_verb be", "insert_verb was",
                            "insert_verb were", "insert_verb will", "insert_verb shell", "insert_verb being",
                            "insert_verb do", "insert_verb does", "insert_verb did", "insert_verb doing",
                            "insert_verb have", "insert_verb has", "insert_verb had", "insert_verb having",
                            "insert_verb can", "insert_verb could", "insert_verb may"]
        inserted_list = ["it", "he", "she", "I", "we", "you", "they", "its", "his", "her", "my", "our",
                         "your", "their", "him", "me", "us", "them", "this", "these", "that", "those",
                         "who", "whom", "which", "ones", "one"]
        inserted_list_prep = ["in", "on", "at", "of", "to", "for", "from", "by", "with", "about", "off", "down", "up",
                              "upon", "within", "above", "below"]
        inserted_list_verb = ["is", "are", "be", "was", "were", "will", "shall", "being", "do", "does", "did", "doing",
                              "have", "has", "had", "having", "can", "could", "may"]
        with open('noun&verb/noun.txt', 'r') as f:
            pairs = f.read().splitlines()
        single_words = [pair.split('\t')[0] for pair in pairs]
        plural_words = [pair.split('\t')[1] for pair in pairs]
        with open('verb/verb_pres_past.txt', 'r') as f:
            pairs = f.read().splitlines()
        verb_present = [pair.split('\t')[0] for pair in pairs]
        verb_past = [pair.split('\t')[1] for pair in pairs]
        with open('verb/verb_full_prdg.txt', 'r') as f:
            quintets = f.read().splitlines()
        verb_single = [pair.split('\t')[1] for pair in quintets]
        verb_plural = [pair.split('\t')[0] for pair in quintets]
        verb_continuous = [triple.split('\t')[2] for triple in quintets]

        d_search = re.compile(r"\w+d\b", re.IGNORECASE)
        ing_search = re.compile(r"\w+ing\b", re.IGNORECASE)
        s_search = re.compile(r"\w+s\b", re.IGNORECASE)
        ed_search = re.compile(r"\w+ed\b", re.IGNORECASE)

        lemmatizer = WordNetLemmatizer()
        positional_symbols = 0

        with open("dict_for_output_competition.json") as json_data_file:
            dict_configs = json.load(json_data_file)

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
                            char = 'THE'
                            prep = self.check_a_an(tokens[tok_place+1].lower())
                            correction_dict = self.replace_multiple([char], prep, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'prep -> the':
                            correction_dict = self.replace_multiple(['A', 'AN'], "the", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'insert a/an':
                            prep = self.check_a_an(word.lower())
                            word_correct = self.upper_or_lower(word, tok_place, prep)
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type == 'insert the':
                            word_correct = self.upper_or_lower(word, tok_place, 'the')
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type == 'delete prep':
                            correct_text = correct_text.replace(word+' ', '')
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type == "plural -> single":
                            inserted_ids = plural_words.index(word.lower())
                            word_correct = single_words[inserted_ids]
                            if word.isupper():
                                word_correct = word_correct.upper()
                            elif word.istitle():
                                word_correct = word_correct.title()
                            if tokens[tok_place - 1] not in ['the', 'The', 'THE']:
                                prep = self.check_a_an(word.lower())
                                word_correct_with_prep = self.upper_or_lower(word_correct, tok_place, prep)
                            correct_text = correct_text.replace(word, word_correct_with_prep)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type == "single -> plural":
                            inserted_ids = single_words.index(word)
                            word_correct = plural_words[inserted_ids]
                            if tokens[tok_place - 1] in ['a', 'A', 'an', 'An', 'AN']:
                                correct_text = correct_text.replace(tokens[tok_place - 1] + word, word_correct)
                                correction_dict[tok_place] = [tokens[tok_place - 1] + word + "->" + word_correct,
                                                              str(word_error_prob), tok_error_type, positional_symbols,
                                                              len(word.encode("utf8"))]
                            else:
                                correct_text = correct_text.replace(word, word_correct)
                                correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                              tok_error_type, positional_symbols,
                                                              len(word.encode("utf8"))]
                        elif tok_error_type == "plural verb -> single":
                            inserted_ids = verb_plural.index(word.lower())
                            word_correct = verb_single[inserted_ids]
                            if word.isupper():
                                word_correct = word_correct.upper()
                            elif word.istitle():
                                word_correct = word_correct.title()
                            correct_text = correct_text.replace(word, word_correct_with_prep)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols, len(word.encode("utf8"))]
                        elif tok_error_type == "single verb -> plural":
                            inserted_ids = verb_single.index(word.lower())
                            word_correct = verb_plural[inserted_ids]
                            if word.isupper():
                                word_correct = word_correct.upper()
                            elif word.istitle():
                                word_correct = word_correct.title()
                            correct_text = correct_text.replace(word, word_correct_with_prep)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols, len(word.encode("utf8"))]
                        elif tok_error_type == "pres -> pres cont":
                            if word.lower() in verb_single:
                                inserted_ids = verb_single.index(word.lower())
                                prep = 'is'
                            elif word.lower() in verb_plural:
                                inserted_ids = verb_plural.index(word.lower())
                                prep = 'are'
                            word_correct = verb_continuous[inserted_ids]
                            if word.isupper():
                                word_correct = word_correct.upper()
                            elif word.istitle():
                                word_correct = word_correct.title()
                            word_correct = self.upper_or_lower(word_correct, tok_place, prep)
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols, len(word.encode("utf8"))]
                        elif tok_error_type == "pres cont -> pres":
                            prep = tokens[tok_place - 1]
                            inserted_ids = verb_continuous.index(word.lower())
                            if prep.lower() == 'is':
                                word_correct = verb_single[inserted_ids]
                            elif prep.lower() == 'are':
                                word_correct = verb_plural[inserted_ids]
                            if word.isupper():
                                word_correct = word_correct.upper()
                            elif word.istitle():
                                word_correct = word_correct.title()
                            correct_text = correct_text.replace(prep+word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type == 'then -> than':
                            correction_dict = self.replace_multiple(['THEN'], "than", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'than -> then':
                            correction_dict = self.replace_multiple(['THAN'], "then", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'bad -> bed':
                            correction_dict = self.replace_multiple(['BAD'], "bed", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'bed -> bad':
                            correction_dict = self.replace_multiple(['BED'], "bad", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'live -> life':
                            correction_dict = self.replace_multiple(['LIVE'], "life", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'life -> live':
                            correction_dict = self.replace_multiple(['LIFE'], "live", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'head -> had':
                            correction_dict = self.replace_multiple(['HEAD'], "had", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'had -> head':
                            correction_dict = self.replace_multiple(['HAD'], "head", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["they -> the",  "the -> they"]:
                            inserted_ids = ["they -> the",  "the -> they"].index(tok_error_type)
                            inserted_l = ["the", "they"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'career -> carrier':
                            correction_dict = self.replace_multiple(['CAREER'], "carrier", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'carrier -> career':
                            correction_dict = self.replace_multiple(['CARRIER'], "career", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'quantity -> a lot of':
                            correction_dict = self.replace_multiple(['MANY', 'MUCH'], "a lot of", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'quantity -> many':
                            correction_dict = self.replace_multiple(['A LOT OF', 'MUCH'], "many", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'quantity -> much':
                            correction_dict = self.replace_multiple(['A LOT OF', 'MANY'], "much", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ['inonatofby -> in', 'inonatofby -> on', 'inonatofby -> at',
                                                'inonatofby -> of', 'inonatofby -> by']:
                            inserted_ids = ['inonatofby -> in', 'inonatofby -> on', 'inonatofby -> at',
                                            'inonatofby -> of', 'inonatofby -> by'].index(tok_error_type)
                            inserted_l = ["in", "on", "at", "of", "by"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == "insert more":
                            word_correct = self.upper_or_lower(word, tok_place, 'more')
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type in ["toforoffrom -> to", "toforoffrom -> for", "toforoffrom -> of",
                                                "toforoffrom -> from"]:
                            inserted_ids = ["toforoffrom -> to", "toforoffrom -> for", "toforoffrom -> of",
                                            "toforoffrom -> from"].index(tok_error_type)
                            inserted_l = ["to", "for", "of", "from"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == 'inwithin -> in':
                            correction_dict = self.replace_multiple(['WITHIN'], "in", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'inwithin -> within':
                            correction_dict = self.replace_multiple(['IN'], "within", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'between -> among':
                            correction_dict = self.replace_multiple(['BETWEEN'], "among", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'among -> between':
                            correction_dict = self.replace_multiple(['AMONG'], "between", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ['thatwhichwhowhom -> that', 'thatwhichwhowhom -> which',
                                                'thatwhichwhowhom -> who', 'thatwhichwhowhom -> whom']:
                            inserted_ids = ['thatwhichwhowhom -> that', 'thatwhichwhowhom -> which',
                                            'thatwhichwhowhom -> who', 'thatwhichwhowhom -> whom'].index(tok_error_type)
                            inserted_l = ["that", "which", "who", "whom"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == 'that -> but':
                            correction_dict = self.replace_multiple(['THAT'], "but", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == 'thisthese -> this':
                            correction_dict = self.replace_multiple(['THESE'], "this", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'thisthese -> these':
                            correction_dict = self.replace_multiple(['THIS'], "these", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)

                        elif tok_error_type == 'thatthose -> that':
                            correction_dict = self.replace_multiple(['THOSE'], "that", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'thatthose -> those':
                            correction_dict = self.replace_multiple(['THAT'], "those", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'where -> there':
                            correction_dict = self.replace_multiple(['WHERE'], "there", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'there -> where':
                            correction_dict = self.replace_multiple(['THERE'], "where", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'this -> it':
                            correction_dict = self.replace_multiple(['THIS'], "it", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'it -> this':
                            correction_dict = self.replace_multiple(['IT'], "this", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'that -> it':
                            correction_dict = self.replace_multiple(['THAT'], "it", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'it -> that':
                            correction_dict = self.replace_multiple(['IT'], "that", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'they -> it':
                            correction_dict = self.replace_multiple(['THEY'], "it", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'it -> they':
                            correction_dict = self.replace_multiple(['IT'], "they", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'them -> it':
                            correction_dict = self.replace_multiple(['THEM'], "it", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'it -> them':
                            correction_dict = self.replace_multiple(['IT'], "them", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["he -> it", "it -> he"]:
                            inserted_ids = ["he -> it", "it -> he"].index(tok_error_type)
                            inserted_l = ["it", "he"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["him -> it", "it -> him"]:
                            inserted_ids = ["him -> it", "it -> him"].index(tok_error_type)
                            inserted_l = ["it", "him"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["she -> it", "it -> she"]:
                            inserted_ids = ["she -> it", "it -> she"].index(tok_error_type)
                            inserted_l = ["it", "she"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["her -> it", "it -> her"]:
                            inserted_ids = ["her -> it", "it -> her"].index(tok_error_type)
                            inserted_l = ["it", "her"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["they -> he", "they -> she", "heshe -> they"]:
                            inserted_ids = ["they -> he", "they -> she", "heshe -> they"].index(tok_error_type)
                            inserted_l = ["he", "she", "they"]
                            inserted_char = inserted_l[inserted_ids]
                            if inserted_ids in [0, 1]:
                                inserted_l = ["they"]
                            else:
                                inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == 'while -> when':
                            correction_dict = self.replace_multiple(['WHILE'], "when", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'when -> while':
                            correction_dict = self.replace_multiple(['WHEN'], "while", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'quantity_pron -> some':
                            correction_dict = self.replace_multiple(['ANY', 'EVERY'], "some", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'quantity_pron -> any':
                            correction_dict = self.replace_multiple(['SOME', 'EVERY'], "any", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'quantity_pron -> every':
                            correction_dict = self.replace_multiple(['SOME', 'ANY'], "every", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'place_pron -> somewhere':
                            correction_dict = self.replace_multiple(['ANYWHERE', 'EVERYWHERE'], "somewhere", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == 'place_pron -> anywhere':
                            correction_dict = self.replace_multiple(['SOMEWHERE', 'EVERYWHERE'], "anywhere", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == 'place_pron -> everywhere':
                            correction_dict = self.replace_multiple(['SOMEWHERE', 'ANYWHERE'], "everywhere", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == 'thing_pron -> something':
                            correction_dict = self.replace_multiple(['ANYTHING', 'EVERYTHING'], "something", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == 'thing_pron -> anything':
                            correction_dict = self.replace_multiple(['SOMETHING', 'EVERYTHING'], "anything", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == 'thing_pron -> everything':
                            correction_dict = self.replace_multiple(['SOMETHING', 'ANYTHING'], "everything", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ['person_pron -> somebody', 'person_pron -> anybody',
                                                'person_pron -> everybody']:
                            inserted_ids = ['person_pron -> somebody', 'person_pron -> anybody',
                                            'person_pron -> everybody'].index(tok_error_type)
                            inserted_l = ["somebody", "anybody", "everybody"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ['person_one -> someone', 'person_one -> anyone',
                                                'person_one -> everyone']:
                            inserted_ids = ['person_one -> someone', 'person_one -> anyone', 'person_one -> everyone'].\
                                index(tok_error_type)
                            inserted_l = ["someone", "anyone", "everyone"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == 'how_pron -> somehow':
                            correction_dict = self.replace_multiple(['ANYHOW'], "somehow", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == 'how_pron -> anyhow':
                            correction_dict = self.replace_multiple(['SOMEHOW'], "anyhow", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in pron_insert_list:
                            correction_dict = self.multiple_insert(pron_insert_list, tok_error_type, inserted_list,
                                                                   word, tok_place, word_error_prob, correction_dict,
                                                                   correct_text, positional_symbols)
                        elif tok_error_type in prep_insert_list:
                            correction_dict = self.multiple_insert(prep_insert_list, tok_error_type, inserted_list_prep,
                                                                   word, tok_place, word_error_prob, correction_dict,
                                                                   correct_text, positional_symbols)
                        elif tok_error_type in verb_insert_list:
                            correction_dict = self.multiple_insert(verb_insert_list, tok_error_type, inserted_list_verb,
                                                                   word, tok_place, word_error_prob, correction_dict,
                                                                   correct_text, positional_symbols)
                        elif tok_error_type in ["count_pron -> one", "count_pron -> ones"]:
                            inserted_ids = ["count_pron -> one", "count_pron -> ones"].index(tok_error_type)
                            inserted_l = ["one", "ones"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["poss_pron -> ones", "poss_pron -> one's"]:
                            inserted_ids = ["poss_pron -> ones", "poss_pron -> one's"].index(tok_error_type)
                            inserted_l = ["one", "one's"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["count_pron -> other", "count_pron -> others"]:
                            inserted_ids = ["count_pron -> other", "count_pron -> others"].index(tok_error_type)
                            inserted_l = ["other", "others"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["pron -> other", "pron -> another", "pron -> different",
                                                "pron -> various"]:
                            inserted_ids = ["pron -> other", "pron -> another", "pron -> different", "pron -> various"]\
                                .index(tok_error_type)
                            inserted_l = ["other", "anothers", "different", "various"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["possibility -> may", "possibility -> might", "possibility -> can",
                                                "possibility -> could"]:
                            inserted_ids = ["possibility -> may", "possibility -> might", "possibility -> can",
                                            "possibility -> could"].index(tok_error_type)
                            inserted_l = ["may", "might", "can", "could"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["would -> will", "will -> would"]:
                            inserted_ids = ["would -> will", "will -> would"].index(tok_error_type)
                            inserted_l = ["will", "would"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == "yet -> already":
                            correction_dict = self.replace_multiple(['YET'], "already", word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["poss_pron -> my", "poss_pron -> mine"]:
                            inserted_ids = ["poss_pron -> my", "poss_pron -> mine"].index(tok_error_type)
                            inserted_l = ["my", "mine"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["poss_pron -> your", "poss_pron -> yours"]:
                            inserted_ids = ["poss_pron -> your", "poss_pron -> yours"].index(tok_error_type)
                            inserted_l = ["your", "yours"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["poss_pron -> her", "poss_pron -> hers"]:
                            inserted_ids = ["poss_pron -> her", "poss_pron -> hers"].index(tok_error_type)
                            inserted_l = ["her", "hers"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["poss_pron -> their", "poss_pron -> theirs"]:
                            inserted_ids = ["poss_pron -> their", "poss_pron -> theirs"].index(tok_error_type)
                            inserted_l = ["their", "theirs"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["self -> I", "self -> me", "self -> myself"]:
                            inserted_ids = ["self -> I", "self -> me", "self -> myself"].index(tok_error_type)
                            inserted_l = ["I", "me", "myself"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["self -> he", "self -> him", "self -> himself"]:
                            inserted_ids = ["self -> he", "self -> him", "self -> himself"].index(tok_error_type)
                            inserted_l = ["he", "him", "himself"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["self -> she", "self -> her", "self -> herself"]:
                            inserted_ids = ["self -> she", "self -> her", "self -> herself"].index(tok_error_type)
                            inserted_l = ["she", "her", "herself"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["self -> you", "self -> yourself", "self -> yourselves"]:
                            inserted_ids = ["self -> you", "self -> yourself", "self -> yourselves"].\
                                index(tok_error_type)
                            inserted_l = ["you", "yourself", "yourselves"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["self -> we", "self -> us", "self -> ourself", "self -> ourselves"]:
                            inserted_ids = ["self -> we", "self -> us", "self -> ourself", "self -> ourselves"].\
                                index(tok_error_type)
                            inserted_l = ["we", "us", "ourself", "ourselves"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["self -> they", "self -> them", "self -> themselves"]:
                            inserted_ids = ["self -> they", "self -> them", "self -> themselves"].index(tok_error_type)
                            inserted_l = ["they", "them", "themselves"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["self -> it", "self -> itself"]:
                            inserted_ids = ["self -> it", "self -> itself"].index(tok_error_type)
                            inserted_l = ["it", "itself"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["self -> one", "self -> oneself"]:
                            inserted_ids = ["self -> one", "self -> oneself"].index(tok_error_type)
                            inserted_l = ["one", "oneself"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == "apos -> s'":
                            apostrof_search = re.compile("'s", re.IGNORECASE)
                            word_correct = apostrof_search.sub("s'", word)
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type == "apos -> 's":
                            apostrof_search = re.compile("s'", re.IGNORECASE)
                            word_correct = apostrof_search.sub("'s", word)
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type in ["apos -> simple", "simple -> apos"]:
                            apostrof_search = re.compile("('s)|(s')", re.IGNORECASE)
                            inserted_ids = ["apos -> simple", "simple -> apos"].index(tok_error_type)
                            if inserted_ids == 0:
                                word_correct = apostrof_search.sub("s", word)
                            else:
                                if word.isupper():
                                    word_correct = word+"'S"
                                else:
                                    word_correct = word + "'s"
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type in ["because of -> due to", "due to -> because of"]:
                            inserted_ids = ["because of -> due to", "due to -> because of"].index(tok_error_type)
                            inserted_l = ["because of", "due to"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word +
                                                                    tokens[tok_place + 1], correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == "in order to -> to":
                            if word.isupper():
                                word_correct = "TO"
                                word = "IN ORDER TO"
                                correct_text = correct_text.replace("IN ORDER TO", word_correct)
                            elif word.istitle():
                                word_correct = "To"
                                word = "In order to"
                                correct_text = correct_text.replace("In order to", word_correct)
                            else:
                                word_correct = "to"
                                word = "in order to"
                                correct_text = correct_text.replace("in order to", word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols - 1 -
                                                          len(tokens[tok_place - 1].encode("utf8")),
                                                          len(word.encode("utf8"))]
                        elif tok_error_type == "past -> did":
                            inserted_ids = verb_past.index(word.lower())
                            word_correct = verb_present[inserted_ids]
                            if word.isupper():
                                word_correct = word_correct.upper()
                            elif word.istitle():
                                word_correct = word_correct.title()
                            word_correct = self.upper_or_lower(word_correct, tok_place, 'did')
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols, len(word.encode("utf8"))]
                        elif tok_error_type == "past -> v3":
                            inserted_ids = verb_present.index(word.lower())
                            word_correct = verb_past[inserted_ids]
                            if word.isupper():
                                word_correct = word_correct.upper()
                                correct_text = correct_text.replace('DID' + word, word_correct)
                            else:
                                correct_text = correct_text.replace('Did' + word, word_correct)
                                correct_text = correct_text.replace('did' + word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols - 1 -
                                                          len(tokens[tok_place - 1].encode("utf8")),
                                                          len(word.encode("utf8"))]
                        elif tok_error_type in ["rather -> more", "more -> rather"]:
                            inserted_ids = ["rather -> more", "more -> rather"].index(tok_error_type)
                            inserted_l = ["rather", "more"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == "unnec -> not nec":
                            correction_dict = self.replace_multiple(['UNNECESSARY'], "not necessary", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["were -> was", "was -> were"]:
                            inserted_ids = ["were -> was", "was -> were"].index(tok_error_type)
                            inserted_l = ["was", "were"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["are -> is", "is -> are"]:
                            inserted_ids = ["are -> is", "is -> are"].index(tok_error_type)
                            inserted_l = ["is", "are"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["have -> has", "has -> have"]:
                            inserted_ids = ["have -> has", "has -> have"].index(tok_error_type)
                            inserted_l = ["has", "have"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["do -> does", "does -> do"]:
                            inserted_ids = ["do -> does", "does -> do"].index(tok_error_type)
                            inserted_l = ["does", "do"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["litter -> liter", "litters -> liters"]:
                            if s_search.search(word) is not None:
                                mistaken = ['LITTERS', "liters"]
                            else:
                                mistaken = ['LITTER', "liter"]
                                correction_dict = self.replace_multiple([mistaken[0]], mistaken[1], word,
                                                                        correction_dict, tok_place, word_error_prob,
                                                                        correct_text, tok_error_type,
                                                                        positional_symbols)
                        elif tok_error_type == "universe -> universal":
                            correction_dict = self.replace_multiple(['UNIVERSE'], "universal", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == "helpful -> helping":
                            correction_dict = self.replace_multiple(['HELPFUL'], "helping", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["interesting -> interested", "interested -> interesting"]:
                            inserted_ids = ["interesting -> interested", "interested -> interesting"].\
                                index(tok_error_type)
                            inserted_l = ["interested", "interesting"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["creating -> creation", "creation -> creating"]:
                            inserted_ids = ["creating -> creation", "creation -> creating"].index(tok_error_type)
                            inserted_l = ["creation", "creating"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["existence -> existant", "existant -> existence"]:
                            inserted_ids = ["existence -> existant", "existant -> existence"].index(tok_error_type)
                            inserted_l = ["existant", "existence"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == "nearby -> near":
                            correction_dict = self.replace_multiple(['NEARBY'], "near", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["weigh -> weight", "weight -> weigh"]:
                            inserted_ids = ["weighs -> weight", "weights -> weigh"].index(tok_error_type)
                            if s_search.search(word) is not None:
                                inserted_l = ["weights", "weighs"]
                            else:
                                inserted_l = ["weight", "weigh"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["separate -> divide", "divide -> separate"]:
                            inserted_ids = ["separate -> divide", "divide -> separate"].index(tok_error_type)
                            if ed_search.search(word) is not None:
                                inserted_l = ["divided", "separated"]
                            elif ing_search.search(word) is not None:
                                inserted_l = ["dividing", "separating"]
                            else:
                                inserted_l = ["divide", "separate"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["cross-cult -> intercult", "intercult -> cross-cult"]:
                            cross_search = re.compile(r"\bcross-\w+", re.IGNORECASE)
                            inter_search = re.compile(r"\binter\w+", re.IGNORECASE)
                            if cross_search.search(word) is not None:
                                word_correct = cross_search.sub("inter", word)
                            else:
                                word_correct = inter_search.sub("cross-", word)
                            correct_text = correct_text.replace(word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type == "education -> medication":
                            correction_dict = self.replace_multiple(['EDUCATION'], "medication", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["large -> big", "big -> large"]:
                            inserted_ids = ["large -> big", "big -> large"].index(tok_error_type)
                            inserted_l = ["big", "large"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["small -> little", "little -> small"]:
                            inserted_ids = ["small -> little", "little -> small"].index(tok_error_type)
                            inserted_l = ["little", "small"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["ofoff -> of", "ofoff -> off"]:
                            inserted_ids = ["ofoff -> of", "ofoff -> off"].index(tok_error_type)
                            inserted_l = ["of", "off"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["outour -> out", "outour -> our"]:
                            inserted_ids = ["outour -> out", "outour -> our"].index(tok_error_type)
                            inserted_l = ["out", "our"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["in demand -> requested", "requested -> in demand"]:
                            inserted_ids = ["in demand -> requested", "requested -> in demand"].index(tok_error_type)
                            inserted_l = ["requested", "in demand"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            if inserted_ids == 0:
                                correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                        tokens[tok_place - 1] + word, correction_dict,
                                                                        tok_place, word_error_prob, correct_text,
                                                                        tok_error_type, positional_symbols - 1 -
                                                                        len(tokens[tok_place - 1].encode("utf8")))
                            else:
                                correction_dict = self.replace_multiple(inserted_l, inserted_char, word,
                                                                        correction_dict, tok_place, word_error_prob,
                                                                        correct_text, tok_error_type,
                                                                        positional_symbols)
                        elif tok_error_type in ["product -> production", "production -> product"]:
                            inserted_ids = ["product -> production", "production -> product"].index(tok_error_type)
                            if s_search.search(word) is not None:
                                inserted_l = ["productions", "products"]
                            else:
                                inserted_l = ["production", "product"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["ed -> addict", "ed -> addicts", "ed -> addicted"]:
                            inserted_ids = ["ed -> addict", "ed -> addicts", "ed -> addicted"].index(tok_error_type)
                            inserted_l = ["addict", "addicts", "addicted"]
                            inserted_char = inserted_l[inserted_ids]
                            if inserted_ids in [0, 1]:
                                inserted_l.pop(1)
                                inserted_l.pop(0)
                            else:
                                inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["cial -> benefit", "cial -> benefits", "cial -> benefited",
                                                "cial -> beneficial"]:
                            inserted_ids = ["cial -> benefit", "cial -> benefits", "cial -> benefited",
                                            "cial -> beneficial"].index(tok_error_type)
                            inserted_l = ["benefit", "benefits", "benefited", "beneficial"]
                            inserted_char = inserted_l[inserted_ids]
                            if inserted_ids in [0, 1, 2]:
                                inserted_l.pop(2)
                                inserted_l.pop(1)
                                inserted_l.pop(0)
                            else:
                                inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["able -> admire", "able -> admired", "able -> admires",
                                                "able -> admirable"]:
                            inserted_ids = ["able -> admire", "able -> admired", "able -> admires",
                                            "able -> admirable"].index(tok_error_type)
                            inserted_l = ["admire", "admired", "admires", "admirable"]
                            inserted_char = inserted_l[inserted_ids]
                            if inserted_ids in [0, 1, 2]:
                                inserted_l.pop(2)
                                inserted_l.pop(1)
                                inserted_l.pop(0)
                            else:
                                inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char,
                                                                    word, correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["ent -> depend", "ent -> depended", "ent -> depends",
                                                "ent -> dependent"]:
                            inserted_ids = ["ent -> depend", "ent -> depended", "ent -> depends", "ent -> dependent"].\
                                index(tok_error_type)
                            inserted_l = ["depend", "depended", "depends", "dependent"]
                            inserted_char = inserted_l[inserted_ids]
                            if inserted_ids in [0, 1, 2]:
                                inserted_l.pop(2)
                                inserted_l.pop(1)
                                inserted_l.pop(0)
                            else:
                                inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == "dependent_on -> addicted_to":
                            correct_text = correct_text.replace("DEPENDENT ON", "ADDICTED TO").\
                                replace("Dependent on", "Addicted to").replace("dependent on", "addicted to")
                            correction_dict[tok_place] = [word + "->" + "addicted to", str(word_error_prob),
                                                          tok_error_type, positional_symbols,
                                                          len(word.encode("utf8"))]
                        elif tok_error_type in ["sayings -> say", "sayings -> tell", "sayings -> talk"]:
                            inserted_ids = ["sayings -> say", "sayings -> tell", "sayings -> talk"].index(tok_error_type)
                            if d_search.search(word) is not None:
                                inserted_l = ["said", "told", "talked"]
                            elif ing_search.search(word) is not None:
                                inserted_l = ["saying", "telling", "talking"]
                            else:
                                inserted_l = ["say", "tell", "talk"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["learnings -> learn", "learnings -> study", "learnings -> teach"]:
                            inserted_ids = ["learnings -> learn", "learnings -> study", "learnings -> teach"].\
                                index(tok_error_type)
                            if s_search.search(word) is not None:
                                inserted_l = ["learns", "studies", "teaches"]
                            elif ing_search.search(word) is not None:
                                inserted_l = ["learning", "studying", "teaching"]
                            else:
                                inserted_l = ["learn", "study", "teach"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["main -> major", "major -> main"]:
                            inserted_ids = ["main -> major", "major -> main"].index(tok_error_type)
                            inserted_l = ["major", "main"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["genetic -> genetical", "genetical -> genetic"]:
                            inserted_ids = ["genetic -> genetical", "genetical -> genetic"].index(tok_error_type)
                            inserted_l = ["genetical", "genetic"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["just -> only", "only -> just"]:
                            inserted_ids = ["just -> only", "only -> just"].index(tok_error_type)
                            inserted_l = ["only", "just"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["all -> all of", "all of -> all"]:
                            inserted_ids = ["all -> all of", "all of -> all"].index(tok_error_type)
                            if inserted_ids == 0:
                                correction_dict = self.replace_multiple(["ALL"], "all of", word, correction_dict,
                                                                        tok_place, word_error_prob,
                                                                        correct_text, tok_error_type, positional_symbols)
                            else:
                                correct_text = correct_text.replace("ALL OF", "ALL").replace("All of", "All").\
                                    replace("all of", "all")
                                correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                              tok_error_type, positional_symbols,
                                                              len(word.encode("utf8")) + 1 +
                                                              len(tokens[tok_place+1].encode("utf8"))]
                        elif tok_error_type in ["transmit -> infect", "infect -> transmit"]:
                            inserted_ids = ["transmit -> infect", "infect -> transmit"].index(tok_error_type)
                            if ed_search.search(word) is not None:
                                inserted_l = ["infected", "transmitted"]
                            elif ing_search.search(word) is not None:
                                inserted_l = ["infecting", "transmitting"]
                            else:
                                inserted_l =["infect", "transmit"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["-> ongoing", "-> continuing"]:
                            inserted_ids = ["-> ongoing", "-> continuing"].index(tok_error_type)
                            inserted_l = ["ongoing", "continuing"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["continuous -> continuing", "continuing -> continuous"]:
                            inserted_ids = ["continuous -> continuing", "continuing -> continuous"].\
                                index(tok_error_type)
                            inserted_l = ["continuing", "continuous"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["droplet -> drop", "drop -> droplet"]:
                            inserted_ids = ["droplet -> drop", "drop -> droplet"].index(tok_error_type)
                            if s_search.search(word) is not None:
                                inserted_l = ["drops", "droplets"]
                            else:
                                inserted_l = ["drop", "droplet"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["maintain -> observe", "observe -> maintain"]:
                            inserted_ids = ["maintain -> observe", "observe -> maintain"].index(tok_error_type)
                            if ed_search.search(word) is not None:
                                inserted_l = ["observed", "maintained"]
                            elif ing_search.search(word) is not None:
                                inserted_l = ["observing", "maintaining"]
                            elif s_search.search(word) is not None:
                                inserted_l = ["observes", "maintains"]
                            else:
                                inserted_l = ["observe", "maintain"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["serious -> earnest", "earnest -> serious"]:
                            inserted_ids = ["serious -> earnest", "earnest -> serious"].index(tok_error_type)
                            ness_search = re.compile(r"\w+nest\b", re.IGNORECASE)
                            if ness_search.search(word) is not None:
                                inserted_l = ["earnestness", "seriousness"]
                            else:
                                inserted_l = ["earnest", "serious"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["impose/adopt -> provide", "provide -> impose", "provide -> adopt"]:
                            inserted_ids = ["impose/adopt -> provide", "provide -> impose", "provide -> adopt"].\
                                index(tok_error_type)
                            if ed_search.search(word) is not None:
                                inserted_l = ["provided", "imposed", "adopted"]
                            elif ing_search.search(word) is not None:
                                inserted_l = ["providing", "imposing", "adopting"]
                            else:
                                inserted_l = ["provide", "impose", "adopt"]
                            inserted_char = inserted_l[inserted_ids]
                            if inserted_ids in [1, 2]:
                                inserted_l.pop(2)
                                inserted_l.pop(1)
                            else:
                                inserted_l.pop(inserted_ids)
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type == "else -> ales":
                            correction_dict = self.replace_multiple(['ELSE'], "ales", word,
                                                                    correction_dict, tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type in ["eliminate -> defeat", "defeat -> eliminatе"]:
                            inserted_ids = ["eliminate -> defeat", "defeat -> eliminatе"].index(tok_error_type)
                            if ed_search.search(word) is not None:
                                inserted_l =["defeated", "eliminated"]
                            elif ing_search.search(word) is not None:
                                inserted_l = ["defeating", "eliminating"]
                            elif s_search.search(word) is not None:
                                inserted_l = ["defeats", "eliminates"]
                            else:
                                inserted_l = ["defeat", "eliminate"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob, correct_text,
                                                                    tok_error_type, positional_symbols)
                        elif tok_error_type in ["right -> true", "true -> right"]:
                            inserted_ids = ["right -> true", "true -> right"].index(tok_error_type)
                            inserted_l = ["true", "right"]
                            inserted_char = inserted_l[inserted_ids]
                            inserted_l.pop(inserted_ids)
                            inserted_l = [elem.upper() for elem in inserted_l]
                            correction_dict = self.replace_multiple(inserted_l, inserted_char, word, correction_dict,
                                                                    tok_place, word_error_prob,
                                                                    correct_text, tok_error_type, positional_symbols)
                        elif tok_error_type == "comparative -> most":
                            word_correct = lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
                            if word.isupper():
                                correct_text = correct_text.replace(word, "MOST" + word_correct.upper())
                            else:
                                correct_text = correct_text.replace(word, "most" + word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols, len(word.encode("utf8"))]
                        elif tok_error_type == "comparative -> more":
                            word_correct = lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
                            if word.isupper():
                                correct_text = correct_text.replace(word, "MORE" + word_correct)
                            else:
                                correct_text = correct_text.replace(word, "more" + word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols, len(word.encode("utf8"))]
                        elif tok_error_type == "comparative -> least":
                            word_correct = lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
                            if word.isupper():
                                correct_text = correct_text.replace(word, "LEAST" + word_correct.upper())
                            else:
                                correct_text = correct_text.replace(word, "least" + word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols, len(word.encode("utf8"))]
                        elif tok_error_type == "comparative -> less":
                            word_correct = lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
                            if word.isupper():
                                correct_text = correct_text.replace(word, "LESS" + word_correct)
                            else:
                                correct_text = correct_text.replace(word, "less" + word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols, len(word.encode("utf8"))]
                        elif tok_error_type in ["most/more_comparative", "least/less_comparative"]:
                            prep = tokens[tok_place-1]  # possibly most or more
                            word_correct = self.choose_comparative_form(word.lower(), prep.lower())
                            if prep.isupper():
                                correct_text = correct_text.replace(prep + word, word_correct.upper())
                            else:
                                correct_text = correct_text.replace(prep + word, word_correct)
                            correction_dict[tok_place] = [word + "->" + word_correct, str(word_error_prob),
                                                          tok_error_type, positional_symbols, len(word.encode("utf8"))]
                        corrected_words.append(word_correct)
                        positional_symbols += len(word.encode("utf8")) + 1  # +1 for space

        return correct_text, correction_dict, words_error, incorrect_words, corrected_words
