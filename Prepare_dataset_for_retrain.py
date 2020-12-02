import pandas as pd
import re
import json
from transformers import BertTokenizer, AutoTokenizer
from Model import GetIndices, TsyaModel
from OldDataPreprocess import DataPreprocess, to_train_val_test_hdf
from transformers.tokenization_bert import BasicTokenizer
from tqdm import tqdm

# parsed_sents = pd.read_excel('ruwiki_2018_09_25_answered_fl_correct_test.xlsx', index_col=0)
# parsed_sents = parsed_sents.rename(columns={"Столбец1": "code"})
# del parsed_sents['Столбец2']
# parsed_sents.to_csv("ruwiki_2018_09_25_answered_with_code.csv", index_label=0)


def write_in_conll_multiple_sent(data, multiplier_code1=3, multiplier_code2=3, multiplier_code3=3, multiplier_code4=2,
                                 suffix=None):
    data = data[~data.corrected.isnull()]
    pattern_n_cased = re.compile(r'(?<=[аоэеиыуёюя])(?-i:Н)'
                                 r'(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
                                 re.IGNORECASE)
    pattern_n = re.compile(r'(?<=[аоэеиыуёюя])'
                           r'(?-i:н)(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
                           re.IGNORECASE)
    pattern_nn_cased = re.compile(r'(?-i:НН)'
                                  r'(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
                                  re.IGNORECASE)
    pattern_nn = re.compile(r'(?-i:нн)'
                            r'(?=([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b)',
                            re.IGNORECASE)
    nn_search = re.compile(r'\wнн([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|'
                           r'ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b',
                           re.IGNORECASE)  # the words, which contain "н" in the middle or in the end of word
    n_search = re.compile(r'[аоэеиыуёюя]н([аоыяеи]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых|ое|ою|ий|его|'
                          r'ему|ем|им|яя|ей|ею|юю|ие|ими|их|ее)\b',
                          re.IGNORECASE)
    tokenizer = BasicTokenizer(do_lower_case=False)
    tokens = []
    labels = []
    for _, sentence in tqdm(data.iterrows()):
        sent_tokens = tokenizer.tokenize(sentence.proc_sentence)
        sent_tokens_corrected = tokenizer.tokenize(sentence.corrected)
        sent_labels = ['O'] * len(sent_tokens)
        sent_labels_mistakes = ['O'] * len(sent_tokens)

        if sentence.code == '!=' or sentence.code == '!':  # mistake of model
            for i, word in enumerate(sent_tokens):
                corrected_word = sent_tokens_corrected[i]
                if word != corrected_word:
                    # print(word, corrected_word)
                    if 'тся' in word:
                        sent_labels_mistakes[i] = 'REPLACE_tsya'
                    elif 'ться' in word:
                        sent_labels_mistakes[i] = 'REPLACE_tysya'
                    else:
                        if len(word) > len(corrected_word):
                            sent_labels_mistakes[i] = 'REPLACE_nn'
                        else:
                            sent_labels_mistakes[i] = 'REPLACE_n'

            for _ in range(multiplier_code1):
                tokens.append(sent_tokens)
                tokens.append(sent_tokens_corrected)
                labels.append(sent_labels)
                labels.append(sent_labels_mistakes)

        elif sentence.code == '2':  # both variants are correct

            for _ in range(multiplier_code2):
                tokens.append(sent_tokens)
                tokens.append(sent_tokens_corrected)
                labels.append(sent_labels)
                labels.append(sent_labels_mistakes)

        elif sentence.code == '1':  # model is correct
            for i, word in enumerate(sent_tokens):
                corrected_word = sent_tokens_corrected[i]
                if word != corrected_word:
                    # print(word, corrected_word)
                    if 'тся' in word:
                        sent_labels_mistakes[i] = 'REPLACE_tysya'
                    elif 'ться' in word:
                        sent_labels_mistakes[i] = 'REPLACE_tsya'
                    else:
                        if len(word) > len(corrected_word):
                            sent_labels_mistakes[i] = 'REPLACE_n'
                        else:
                            sent_labels_mistakes[i] = 'REPLACE_nn'

            for _ in range(multiplier_code3):
                tokens.append(sent_tokens)
                tokens.append(sent_tokens_corrected)
                labels.append(sent_labels_mistakes)
                labels.append(sent_labels)

    with open("all_dliNa_sent.txt", 'r') as f_dlin:
        dlin_sentences = f_dlin.read().splitlines()
        for line in dlin_sentences:
            sent_tokens = tokenizer.tokenize(line.strip())
            sent_tokens_corrected = sent_tokens
            sent_labels = ['O'] * len(sent_tokens)
            sent_labels_mistakes = ['O'] * len(sent_tokens)
            for i, token in enumerate(sent_tokens):
                if nn_search.search(token) is not None:
                    sent_labels_mistakes[i] = 'REPLACE_nn'
                    sent_tokens_corrected[i] = pattern_nn_cased.sub('Н', token)
                    sent_tokens_corrected[i] = pattern_nn.sub('н', sent_tokens_corrected[i])
                elif n_search.search(token) is not None:
                    sent_labels_mistakes[i] = 'REPLACE_n'
                    sent_tokens_corrected[i] = pattern_n_cased.sub('НН', token)
                    sent_tokens_corrected[i] = pattern_n.sub('нн', sent_tokens_corrected[i])

            for _ in range(multiplier_code4):
                tokens.append(sent_tokens)
                tokens.append(sent_tokens_corrected)
                labels.append(sent_labels)
                labels.append(sent_labels_mistakes)

    with open("datasets/"+"dataset_for_retrain"+suffix+".txt", 'w') as file:
        zipped = list(zip(tokens, labels))
        for i in range(len(zipped)):
            sentence_tokens = zipped[i][0]
            sentence_labels = zipped[i][1]
            for j in range(len(sentence_tokens)):
                file.write(sentence_tokens[j] + '\t' + sentence_labels[j] + '\n')
            file.write('\n')


def write_multiple_sentences(data, multiplier_code1=3, multiplier_code2=3, multiplier_code3=3, suffix=None):
    data = data[~data.corrected.isnull()]
    with open("dataset_for_retrain"+suffix+".txt", 'w') as f:
        for _, sentence in data.iterrows():
            if sentence.code == '!=' or sentence.code == '!':
                for _ in range(multiplier_code1):
                    f.write(sentence.proc_sentence)
                    f.write("\n")
            elif sentence.code == '2':
                for _ in range(multiplier_code2):
                    f.write(sentence.proc_sentence)
                    f.write("\n")
                    f.write(sentence.corrected)
                    f.write("\n")
            elif sentence.code == '1':
                for _ in range(multiplier_code3):
                    f.write(sentence.corrected)
                    f.write("\n")

        with open("all_dliNa_sent.txt", 'r') as f_dlin:
            dlin_sentences = f_dlin.read().splitlines()
            for _ in range(multiplier_code2):
                for line in dlin_sentences:
                    f.write(line.strip())
                    f.write("\n")


def create_test_file_conll(suffix):

    tokens = []
    labels = []
    tokenizer = BasicTokenizer(do_lower_case=False)
    with open("dataset_for_retrain"+suffix+".txt", 'r') as f:
        sentences = f.read().splitlines()
    sentences = [line.strip() for line in sentences]

    for i in tqdm(range(len(sentences))):
        row = sentences[i]
        sent_tokens = tokenizer.tokenize(row)
        sent_labels = ['O'] * len(sent_tokens)
        tokens.append(sent_tokens)
        labels.append(sent_labels)

    with open("conll" + "dataset_for_retrain"+suffix+".txt", 'w') as f:
        zipped = list(zip(tokens, labels))
        for i in range(len(zipped)):
            sentence_tokens = zipped[i][0]
            sentence_labels = zipped[i][1]
            for j in range(len(sentence_tokens)):
                f.write(sentence_tokens[j] + '\t' + sentence_labels[j] + '\n')
            f.write('\n')


def main(create_file=False, test_indices=True, retrain=True, suffix="_3333_with_mistakes_prob_0.5_full"):
    if create_file:
        parsed_sents = pd.read_csv("ruwiki_answered/"+"ruwiki_2018_09_25_answered_with_code.csv", index_col=0)
        print(parsed_sents.head())
        write_multiple_sentences(parsed_sents, suffix=suffix)
        # write_in_conll_multiple_sent(parsed_sents, suffix=suffix)
    if test_indices:
        # create_test_file_conll(suffix)
        with open("config_train.json") as json_data_file:
            configs = json.load(json_data_file)

        if not configs['from_rubert']:
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

        else:
            tokenizer = AutoTokenizer.from_pretrained(**configs['config_of_tokenizer'])

        # data_processor = DataPreprocess(path_to_file="datasets/"+"dataset_for_retrain" + suffix + ".txt",
        #                                 label_list=configs["label_list"],
        #                                 tokenizer=tokenizer)
        # data_processor.process_batch(output_file='ids_test_fl' + suffix + '.hdf5', data_dir="",
        #                              part_of_word=False, file_size=8000)
        to_train_val_test_hdf(data_dir="./", output_dir="./",
                              train_part=1.0, val_part=0.0,
                              length=13232, random_seed=1,
                              use_both_datasets=False, filename='ids_test_fl' + suffix + '.hdf5',
                              suffix=suffix)

    if retrain:

        with open("config_train.json") as json_data_file:
            configs = json.load(json_data_file)

        if not configs['from_rubert']:
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

        else:
            tokenizer = AutoTokenizer.from_pretrained(**configs['config_of_tokenizer'])

        train_data_processor = GetIndices(ftype=configs["train_file"], data_dir="")
        train_data_processor.upload_hdf()
        val_data_processor = GetIndices(ftype=configs["val_file"], data_dir="")
        val_data_processor.upload_hdf()

        model = TsyaModel(label_list=configs["label_list"], weight_path=configs["weight_path"]+configs["chckp_file"],
                          train_from_chk=configs["train_from_chk"],
                          seed_val=configs['seed_val'], tokenizer=tokenizer, from_rubert=configs['from_rubert'],
                          config_of_model=configs['config_of_model'], adam_options=configs['adam_options'])
        model.train(train_data_processor=train_data_processor, val_data_processor=val_data_processor,
                    chkp_path="Chkpts/Chkpt_fl_hardsoft_correct_2set_retrained"+suffix+".pth", epochs=configs["epochs"],
                    batch_size=configs["batch_size"])


if __name__ == '__main__':
    main()

