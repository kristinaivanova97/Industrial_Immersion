import pandas as pd
import re
from Model import GetIndices, TsyaModel
# from DataPreprocess import DataPreprocess, to_train_val_test_hdf
from tqdm import tqdm
# from Get_json_readable import sentence_tokenize, sent_tokenize, get_files
# from Class_for_execution import OrphoNet, TestPreprocess, ProcessOutput, TsyaModel
import numpy as np
from Class import TestPreprocess, ProcessOutput
import json
from transformers import BertTokenizer
from transformers.tokenization_bert import BasicTokenizer


def add_sentences_to_retrain(df):
    model = OrphoNet()
    retrain_sent = np.array([])
    with open("retrain_readable_2.txt", 'w') as f:
        for i, row in tqdm(df.iterrows()):
            line = row.sentence
            if line is not np.nan:
                output = model.execute(line)
                if len(output) > 1 and output[0] == 'Incorrect':
                    retrain_sent = np.append(retrain_sent, line)
                    f.write(line)
                    f.write('\n')
                    f.write(line)
                    f.write('\n')

    return retrain_sent


def write_dataset_to_file(tokens, labels, file):
    with open(file, 'w') as f:
        zipped = list(zip(tokens, labels))
        for l in range(len(zipped)):
            sentence_tokens = zipped[l][0]
            sentence_labels = zipped[l][1]
            # print(sentence_tokens, sentence_labels, len(sentence_tokens))
            for k in range(len(sentence_tokens)):
                f.write(sentence_tokens[k] + '\t' + sentence_labels[k] + '\n')
            f.write('\n')


def multiple_tags_correct(labels, word, tok_error_type, correct_text, positional_symbols, word_correct,
                          mistake_list, word_to_replace, tag_list):
    inserted_ids = mistake_list.index(tok_error_type)
    inserted_l = word_to_replace.copy()
    inserted_char = inserted_l[inserted_ids]
    inserted_l.pop(inserted_ids)
    inserted_l = [elem.upper() for elem in inserted_l]
    for char in inserted_l:
        if len(inserted_char) > 1:
            if len(char) > 1:
                word_correct = word.replace(char, inserted_char.upper()). \
                    replace(char[0] + char[1:].lower(),
                            inserted_char[0].upper() + inserted_char[1:]). \
                    replace(char.lower(), inserted_char)
            else:
                word_correct = word.replace(char, inserted_char.upper()). \
                    replace(char, inserted_char[0].upper() + inserted_char[1:]). \
                    replace(char.lower(), inserted_char)
        else:
            if len(char) > 1:
                word_correct = word.replace(char, inserted_char.upper()). \
                    replace(char[0] + char[1:].lower(), inserted_char.upper()). \
                    replace(char.lower(), inserted_char)
            else:
                word_correct = word.replace(char, inserted_char.upper()). \
                    replace(char[0], inserted_char.upper()). \
                    replace(char.lower(), inserted_char)
        if word != word_correct:
            break
    replace_text = correct_text[positional_symbols:positional_symbols + len(word.encode("utf8"))
                                                   + 1].replace(word, word_correct)
    correct_text = "".join((correct_text[: positional_symbols], replace_text,
                            correct_text[positional_symbols + len(word.encode("utf8")) + 1:]))
    ids = word_to_replace.index(word.lower())
    labels.append(tag_list[ids])
    return word, word_correct, correct_text, labels


def clever_retrain_set(df):

    basic_tokenizer = BasicTokenizer(do_lower_case=False)
    with open("config_stand.json") as json_data_file:
        configs = json.load(json_data_file)
    with open("test.json") as json_data_file:
        tags = json.load(json_data_file)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    output = ProcessOutput(tokenizer=tokenizer)
    data_processor = TestPreprocess(tokenizer=tokenizer)
    model = TsyaModel(weight_path=configs['weight_path'] + configs['chckp_file'],
                      train_from_chk=configs['train_from_chk'], label_list=configs['label_list'],
                      seed_val=configs["seed_val"], from_rubert=configs['from_rubert'],
                      adam_options=configs["adam_options"], tokenizer=tokenizer,
                      config_of_model=configs["config_of_model"], multilingual=False)
    all_labels = []
    all_words = []
    df = df[:300]
    for j, row in tqdm(df.iterrows()):
        replace_sentence = False
        sentence = row.proc_sentence
        input_ids, mask_ids, prediction_dataloader, nopad = data_processor.process(text=[sentence])
        predicts, probabilities, probabilities_o = model.predict_batch(prediction_dataloader, nopad)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0, :nopad[0]])
        preds = np.array(predicts[0][0])
        replace_lists = output.get_replace_lists(preds)
        threshold = 0.5
        words = []
        corrected_words = []
        correct_text = sentence
        labels = []
        symb = 0
        while sentence[symb] == ' ':
            symb += 1
        tabs = 0
        while sentence[0][tabs] == '\t':
            tabs += 1
        backslash_n = 0
        while sentence[0][backslash_n] == '\n':
            backslash_n += 1
        positional_symbols = int(0) - len("[CLS]".encode("utf8")) + symb + tabs * len('\t'.encode("utf8")) + \
                             backslash_n * len('\n'.encode("utf8"))
        try:
            for tok_place, token in enumerate(tokens):
                tok_replace_probs = []
                tok_hold_probs = []
                word = token
                k = 1
                if '##' not in token:
                    tok_error_type = output.check_contain_mistakes(tok_place, probabilities[tok_place],
                                                                   probabilities_o[tok_place], replace_lists, threshold,
                                                                   tok_replace_probs, tok_hold_probs)

                    if tok_place + k < len(tokens):
                        while '##' in tokens[tok_place + k]:
                            word += tokens[tok_place + k][2:]
                            tok_error_type = output.check_contain_mistakes(tok_place, probabilities[tok_place],
                                                                         probabilities_o[tok_place], replace_lists,
                                                                         threshold, tok_replace_probs, tok_hold_probs)
                            k += 1
                    next_tok_place = tok_place + k
                    words.append(word)
                    word_correct = ''
                    word_with_prep = []
                    if len(tok_replace_probs) > 0:

                        if tok_error_type == 'prep -> a/an':
                            prep = 'a'
                            if len(tokens[tok_place + 1]) > 1:
                                prep = output.check_a_an(tokens[tok_place + 1].lower())
                            elif tokens[tok_place + 1] == "'":
                                prep = output.check_a_an(tokens[tok_place + 2].lower())
                            else:
                                if tokens[tok_place + 1].isupper():
                                    prep = 'A'
                                else:
                                    prep = 'a'
                            replace_char = prep
                            for char in ['THE']:
                                if len(replace_char) > 1:
                                    if len(char) > 1:
                                        word_correct = word.replace(char, replace_char.upper()). \
                                            replace(char[0] + char[1:].lower(),
                                                    replace_char[0].upper() + replace_char[1:]). \
                                            replace(char.lower(), replace_char)
                                    else:
                                        word_correct = word.replace(char, replace_char.upper()). \
                                            replace(char, replace_char[0].upper() + replace_char[1:]). \
                                            replace(char.lower(), replace_char)
                                else:
                                    if len(char) > 1:
                                        word_correct = word.replace(char, replace_char.upper()). \
                                            replace(char[0] + char[1:].lower(), replace_char.upper()). \
                                            replace(char.lower(), replace_char)
                                    else:
                                        word_correct = word.replace(char, replace_char.upper()). \
                                            replace(char[0], replace_char.upper()). \
                                            replace(char.lower(), replace_char)
                                if word != word_correct:
                                    break
                            replace_text = correct_text[positional_symbols:positional_symbols +
                                                        len(word.encode("utf8")) + 1].replace(word, word_correct)
                            correct_text = "".join((correct_text[: positional_symbols], replace_text,
                                                    correct_text[positional_symbols + len(word.encode("utf8")) + 1:]))
                            labels.append("REPLACE_the")
                        elif tok_error_type == 'prep -> the':
                            replace_char = "the"
                            for char in ['A', 'AN']:
                                if len(replace_char) > 1:
                                    if len(char) > 1:
                                        word_correct = word.replace(char, replace_char.upper()). \
                                            replace(char[0] + char[1:].lower(),
                                                    replace_char[0].upper() + replace_char[1:]). \
                                            replace(char.lower(), replace_char)
                                    else:
                                        word_correct = word.replace(char, replace_char.upper()). \
                                            replace(char, replace_char[0].upper() + replace_char[1:]). \
                                            replace(char.lower(), replace_char)
                                else:
                                    if len(char) > 1:
                                        word_correct = word.replace(char, replace_char.upper()). \
                                            replace(char[0] + char[1:].lower(), replace_char.upper()). \
                                            replace(char.lower(), replace_char)
                                    else:
                                        word_correct = word.replace(char, replace_char.upper()). \
                                            replace(char[0], replace_char.upper()). \
                                            replace(char.lower(), replace_char)
                                if word != word_correct:
                                    break
                            replace_text = correct_text[positional_symbols:positional_symbols +
                                                                           len(word.encode("utf8")) + 1].\
                                replace(word, word_correct)
                            correct_text = "".join((correct_text[: positional_symbols], replace_text,
                                                    correct_text[positional_symbols + len(word.encode("utf8")) + 1:]))
                            labels.append("REPLACE_a_an")
                        elif tok_error_type == 'insert prep a/an':
                            prep = 'a'
                            if len(word) > 1:
                                prep = output.check_a_an(word.lower())
                            elif word == "'":
                                prep = output.check_a_an(tokens[tok_place + 1].lower())
                            else:
                                if word.isupper():
                                    prep = 'A'
                                else:
                                    prep = 'a'
                            word_correct = output.upper_or_lower(word, tok_place, prep)
                            word_with_prep = basic_tokenizer.tokenize(word_correct)
                            replace_text = correct_text[positional_symbols:positional_symbols +
                                                                           len(word.encode("utf8")) + 1]. \
                                replace(word, word_correct)
                            correct_text = "".join((correct_text[: positional_symbols], replace_text,
                                                    correct_text[positional_symbols + len(word.encode("utf8")) + 1:]))
                            labels.append("DELETE_prep")
                            labels.append("O")
                        elif tok_error_type == 'insert prep the':
                            word_correct = output.upper_or_lower(word, tok_place, 'the')
                            word_with_prep = basic_tokenizer.tokenize(word_correct)
                            replace_text = correct_text[positional_symbols:positional_symbols +
                                                                           len(word.encode("utf8")) + 1]. \
                                replace(word, word_correct)
                            correct_text = "".join((correct_text[: positional_symbols], replace_text,
                                                    correct_text[positional_symbols + len(word.encode("utf8")) + 1:]))
                            labels.append("DELETE_prep")
                            labels.append("O")
                        elif tok_error_type == 'delete prep':
                            if sentence[positional_symbols+1].lower() == 't':
                                labels.append("INSERT_the")
                            else:
                                labels.append("INSERT_a_an")
                        elif tok_error_type in ['inonatofby -> in', 'inonatofby -> on', 'inonatofby -> at',
                                                'inonatofby -> of', 'inonatofby -> by']:
                            word, word_correct, correct_text, labels = \
                                multiple_tags_correct(labels, word, tok_error_type, correct_text, positional_symbols,
                                                      word_correct, ['inonatofby -> in', 'inonatofby -> on',
                                                                     'inonatofby -> at', 'inonatofby -> of',
                                                                     'inonatofby -> by'],
                                                      ["in", "on", "at", "of", "by"],
                                                      ["REPLACE_inonatofby_in", "REPLACE_inonatofby_on",
                                                       "REPLACE_inonatofby_at", "REPLACE_inonatofby_of"])
                        elif tok_error_type in ['inwithin -> in', 'inwithin -> within']:
                            word, word_correct, correct_text, labels =\
                                multiple_tags_correct(labels, word, tok_error_type, correct_text, positional_symbols,
                                                      word_correct, ['inwithin -> in', 'inwithin -> within'],
                                                      ["in", "within"],
                                                      ["REPLACE_inwithin_within", "REPLACE_inwithin_in"])
                        elif tok_error_type in ['thatwhichwhowhom -> that', 'thatwhichwhowhom -> which',
                                                'thatwhichwhowhom -> whom', 'thatwhichwhowhom -> who']:
                            word, word_correct, correct_text, labels = \
                                multiple_tags_correct(labels, word, tok_error_type, correct_text, positional_symbols,
                                                      word_correct, ['thatwhichwhowhom -> that',
                                                                     'thatwhichwhowhom -> which',
                                                                     'thatwhichwhowhom -> whom',
                                                                     'thatwhichwhowhom -> who'],
                                                      ["that", "which", "whom", "who"],
                                                      ["REPLACE_that", "REPLACE_which", "REPLACE_who", "REPLACE_whom"])
                        else:
                            labels.append('O')
                    else:
                        labels.append('O')

                    if len(word_correct) > 0:
                        word = word_correct
                        replace_sentence = True
                        if len(word_with_prep) > 0:
                            corrected_words.append(word_with_prep[0])
                            corrected_words.append(word_with_prep[1])
                        else:
                            corrected_words.append(word_correct)
                    else:
                        corrected_words.append(word)
                    word_count = word
                    if len(word_correct) > 0:
                        word_count = word_correct  # to correctly add position in case of several mistakes
                    s = 0
                    if positional_symbols + len(word_count.encode("utf8")) < len(correct_text):
                        while correct_text[positional_symbols + len(word_count.encode("utf8")) + s] == ' ':
                            s += 1

                    if next_tok_place < len(tokens):
                        if correct_text.find(word_count + ' ' * s + tokens[next_tok_place]) >= 0:
                            positional_symbols += len(word_count.encode("utf8")) + s
                        else:
                            positional_symbols += len(word_count.encode("utf8"))
                    else:
                        positional_symbols += len(word_count.encode("utf8"))
            if replace_sentence:
                all_words.append(words[1:-1])
                all_words.append(corrected_words[1:-1])
                all_labels.append(['O']*len(words[1:-1]))
                all_labels.append(labels[1:-1])
        except IndexError:
            print("MISTAKEN", sentence)
    write_dataset_to_file(all_words, all_labels, "retrain_dataset_test.txt")


def retrain_model(get_ids=True, retrain=False, suffix='_only_locness'):
    if get_ids:
        with open("config_retrain.json") as json_data_file:
            configs = json.load(json_data_file)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        data_processor = DataPreprocess(path_to_file="datasets_readable/"+"dataset_retrain_2_v2.txt",
                                        label_list=configs["label_list"],
                                        tokenizer=tokenizer)
        data_processor.process_batch(output_file='ids_fl' + suffix + '.hdf5', data_dir="",
                                     part_of_word=False, file_size=134)
        to_train_val_test_hdf(data_dir="./", output_dir="./",
                              train_part=1.0, val_part=0.0,
                              length=134, random_seed=1,
                              use_both_datasets=False, filename='ids_fl' + suffix + '.hdf5',
                              suffix=suffix)

    if retrain:

        with open("config_retrain.json") as json_data_file:
            configs = json.load(json_data_file)

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        train_data_processor = GetIndices(ftype=configs["train_file"], data_dir="")
        train_data_processor.upload_hdf()
        val_data_processor = GetIndices(ftype=configs["val_file"], data_dir="")
        val_data_processor.upload_hdf()

        model = TsyaModel(label_list=configs["label_list"], weight_path=configs["weight_path"]+configs["chckp_file"],
                          train_from_chk=configs["train_from_chk"],
                          seed_val=configs['seed_val'], tokenizer=tokenizer, from_rubert=configs['from_rubert'],
                          adam_options=configs['adam_options'], multilingual=False)
        model.train(train_data_processor=train_data_processor, val_data_processor=val_data_processor,
                    chkp_path="Chkpts/Chkpt_fl_100"+suffix+".pth", epochs=configs["epochs"],
                    batch_size=configs["batch_size"])


def main():
    # dir = 'LOCNESS-corpus-files/'
    # df = get_files(dir)
    # print(df)
    # _ = add_sentences_to_retrain(df)
    # retrain_model(get_ids=False, retrain=True, suffix='_only_locness')

    df = pd.read_csv("enwiki_small.csv")
    clever_retrain_set(df)


if __name__ == '__main__':

    main()
