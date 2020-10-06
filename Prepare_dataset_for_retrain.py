import pandas as pd
import re
import json
from transformers import BertTokenizer, AutoTokenizer
from Model import GetIndices, TsyaModel
from DataPreprocess import DataPreprocess, to_train_val_test_hdf
from transformers.tokenization_bert import BasicTokenizer
from tqdm import tqdm

# parsed_sents = pd.read_excel('ruwiki_2018_09_25_answered_fl_correct_test.xlsx', index_col=0)
# parsed_sents = parsed_sents.rename(columns={"Столбец1": "code"})
# del parsed_sents['Столбец2']
# parsed_sents.to_csv("ruwiki_2018_09_25_answered_with_code.csv", index_label=0)


def write_multiple_sentences(data, multiplier_code1=5, multiplier_code2=2, multiplier_code3=5):
    with open("dataset_for_retrain_5252.txt", 'w') as f:
        for _, sentence in data.iterrows():
            if sentence.code == '!=':
                for _ in range(multiplier_code1+1):
                    f.write(sentence.proc_sentence)
                    f.write("\n")
            elif sentence.code == 2:
                for _ in range(multiplier_code2+1):
                    f.write(sentence.proc_sentence)
                    f.write("\n")
                    f.write(sentence.corrected)
                    f.write("\n")
            elif sentence.code == 1 and re.search('длин', sentence):
                for _ in range(multiplier_code3+1):
                    f.write(sentence.proc_sentence)
                    f.write("\n")

        with open("all_dliNa_sent.txt", 'r') as f_dlin:
            dlin_sentences = f_dlin.read().splitlines()
            for _ in range(multiplier_code2+1):
                for line in dlin_sentences:
                    f.write(line.strip())
                    f.write("\n")


def create_test_file_conll(file="dataset_for_retrain_5252.txt"):

    tokens = []
    labels = []
    tokenizer = BasicTokenizer(do_lower_case=False)
    with open(file, 'r') as f:
        sentences = f.read().splitlines()
    sentences = [line.strip() for line in sentences]

    for i in tqdm(range(len(sentences))):
        row = sentences[i]
        sent_tokens = tokenizer.tokenize(row)
        sent_labels = ['O'] * len(sent_tokens)
        tokens.append(sent_tokens)
        labels.append(sent_labels)

    with open("conll" + file, 'w') as f:
        zipped = list(zip(tokens, labels))
        for i in range(len(zipped)):
            sentence_tokens = zipped[i][0]
            sentence_labels = zipped[i][1]
            for j in range(len(sentence_tokens)):
                f.write(sentence_tokens[j] + '\t' + sentence_labels[j] + '\n')
            f.write('\n')


def main(create_file=False, test_indices=False, retrain=True, suffix=None):
    if create_file:
        parsed_sents = pd.read_csv("ruwiki_2018_09_25_answered_with_code.csv", index_col=0)
        print(parsed_sents.head())
        write_multiple_sentences(parsed_sents)
    if test_indices:
        suffix = "_5252."
        create_test_file_conll()
        with open("config_train.json") as json_data_file:
            configs = json.load(json_data_file)

        if not configs['from_rubert']:
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

        else:
            tokenizer = AutoTokenizer.from_pretrained(**configs['config_of_tokenizer'])

        data_processor = DataPreprocess(path_to_file="conlldataset_for_retrain" + suffix + "txt",
                                        label_list=configs["label_list"],
                                        tokenizer=tokenizer)
        data_processor.process_batch(output_file='ids_test_fl' + suffix + 'hdf5', data_dir="",
                                     part_of_word=False, file_size=17117)
        to_train_val_test_hdf(data_dir="./", output_dir="./",
                              train_part=1.0, val_part=0.0,
                              length=17118, random_seed=1,
                              use_both_datasets=False, filename='ids_test_fl' + suffix + 'hdf5')

    if retrain:

        with open("config_train.json") as json_data_file:
            configs = json.load(json_data_file)

        if not configs['from_rubert']:
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

        else:
            tokenizer = AutoTokenizer.from_pretrained(**configs['config_of_tokenizer'])

        train_data_processor = GetIndices(ftype=configs["train_file"], data_dir="./")
        train_data_processor.upload_hdf()
        val_data_processor = GetIndices(ftype=configs["val_file"], data_dir="./")
        val_data_processor.upload_hdf()

        model = TsyaModel(label_list=configs["label_list"], weight_path=configs["weight_path"]+configs["chckp_file"],
                          train_from_chk=configs["train_from_chk"],
                          seed_val=configs['seed_val'], tokenizer=tokenizer, from_rubert=configs['from_rubert'],
                          config_of_model=configs['config_of_model'], adam_options=configs['adam_options'])
        model.train(train_data_processor=train_data_processor, val_data_processor=val_data_processor,
                    chkp_path="Chkpts/Chkpt_fl_hardsoft_correct_2set_retrained"+suffix+"pth", epochs=configs["epochs"],
                    batch_size=configs["batch_size"])


if __name__ == '__main__':
    main()
