import json
import h5py
from transformers import BertTokenizer, AutoTokenizer
from DataPreprocess import DataPreprocess
import random
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def merge_2_dsets(magazines_file, news_file, output_file, len_of_data, percentage):
    dtype_dict = {"input_ids": 'i8', "input_mask": 'i1', "label_ids": 'i1'}
    first = int(len_of_data*percentage)
    print(first)
    if first % 100 != 0:
        first += 1
        print(first)
    second = int(len_of_data * (1-percentage))
    with h5py.File(output_file, 'w') as f:
        with h5py.File(magazines_file, 'r') as fm:
            with h5py.File(news_file, 'r') as fn:
                for ftype in ['input_ids', 'input_mask', 'label_ids']:
                    output_data = f.create_dataset(ftype, (len_of_data, 512),
                                                   maxshape=(1000000, 512),
                                                   dtype=dtype_dict[ftype])
                    output_data[:first] = fm[ftype][:first]
                    output_data[first:len_of_data] = fn[ftype][:second]


def write_dataset_to_file(tokens, labels, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        zipped = list(zip(tokens, labels))
        for i in range(len(zipped)):
            sentence_tokens = zipped[i][0]
            sentence_labels = zipped[i][1]
            for j in range(len(sentence_tokens)):
                f.write(sentence_tokens[j] + '\t' + sentence_labels[j] + '\n')
            f.write('\n')


def create_small_dset(file, num_of_sentences, len_of_train=84000, len_of_val=28000):

    with open(file, 'r', encoding='utf-8') as file:
        line = file.readline()
        line_list = line.strip().split()
        list_of_words = []
        list_of_labels = []
        sentences_train = []
        labels_train = []
        sentences_val = []
        labels_val = []

        idxs = list(range(num_of_sentences))
        random.seed(1)
        random.shuffle(idxs)
        idxs_train = idxs[:len_of_train]
        idxs_val = idxs[len_of_train: len_of_train + len_of_val]

        i = 0
        while line:
            if len(line_list) > 1:
                list_of_words.append(line_list[0])
                list_of_labels.append(line_list[1])
            else:
                if (i % 10000) == 0:
                    print(i)
                if i in idxs_train:
                    sentences_train.append(list_of_words)
                    labels_train.append(list_of_labels)
                if i in idxs_val:
                    sentences_val.append(list_of_words)
                    labels_val.append(list_of_labels)
                list_of_words = []
                list_of_labels = []
                i += 1

            line = file.readline()
            line_list = line.strip().split()

    return sentences_train, labels_train, sentences_val, labels_val


def main(cut_from_dataset=False, make_pleminary_indices=False):

    with open("template_config_datapreprocess.json", "r+") as jsonFile:
        configs = json.load(jsonFile)
    if not configs['from_rubert']:
        tokenizer = BertTokenizer.from_pretrained(**configs['config_of_tokenizer'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(**configs['config_of_tokenizer'])

    num_of_sent = [configs["news_filesize"], configs["magazines_filesize"]]
    universal_txt_dir = configs["universal_txt_dir"]
    universal_hdf_dir = configs["data_universal_path"]
    if cut_from_dataset:
        for j, ftype in enumerate(["news_", ""]):
            sentences_train, labels_train, sentences_val, labels_val = create_small_dset(
                file="./datasets/" + ftype + "dataset_hardsoft_correct.txt", num_of_sentences=num_of_sent[j],
                len_of_train=configs["length_of_data"]*configs["train_part"],
                len_of_val=configs["length_of_data"]*configs["val_part"])
            write_dataset_to_file(tokens=sentences_train, labels=labels_train,
                                  output_file=universal_txt_dir + "./universal_" + ftype + "train.txt")
            write_dataset_to_file(tokens=sentences_val, labels=labels_val,
                                  output_file=universal_txt_dir + "./universal_" + ftype + "val.txt")
            print(ftype, " is done ")

    if make_pleminary_indices:
        label_list_pow = ["[PAD]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya", "REPLACE_tsya",
                          "[##]"]
        lists = [label_list_pow, label_list_pow[:-1]]
        bools = [True, False]
        for i, data_type in enumerate(['pow', 'fl']):
            data_processor = DataPreprocess(path_to_file=universal_txt_dir+"./universal_news_train.txt",
                                            label_list=lists[i], tokenizer=tokenizer)
            data_processor.process_batch(output_file='ids_universal_news_train_' + data_type + '.hdf5',
                                         data_dir=universal_hdf_dir, part_of_word=bools[i],
                                         file_size=configs["length_of_data"]*configs["train_part"])
            data_processor = DataPreprocess(path_to_file=universal_txt_dir+"./universal_news_val.txt",
                                            label_list=lists[i], tokenizer=tokenizer)
            data_processor.process_batch(output_file='ids_universal_news_val_' + data_type + '.hdf5',
                                         data_dir=universal_hdf_dir,
                                         part_of_word=bools[i], file_size=configs["length_of_data"]*configs["val_part"])
            print("Finished with news")
            data_processor = DataPreprocess(path_to_file=universal_txt_dir+"./universal_train.txt", label_list=lists[i],
                                            tokenizer=tokenizer)
            data_processor.process_batch(output_file='ids_universal_magazines_train_' + data_type + '.hdf5',
                                         data_dir=universal_hdf_dir, part_of_word=bools[i],
                                         file_size=configs["length_of_data"]*configs["train_part"])
            data_processor = DataPreprocess(path_to_file=universal_txt_dir+"./universal_val.txt", label_list=lists[i],
                                            tokenizer=tokenizer)
            data_processor.process_batch(output_file='ids_universal_magazines_val_' + data_type + '.hdf5',
                                         data_dir=universal_hdf_dir,
                                         part_of_word=bools[i], file_size=configs["length_of_data"]*configs["val_part"])
            print("processed")
    merge = configs["merge_datasets"]
    data_type = configs["data_type"]
    if merge:
        merge_2_dsets(universal_hdf_dir + data_type + '_magazines/' + 'ids_universal_magazines_train_' + data_type +
                      '.hdf5',
                      universal_hdf_dir + data_type + '_news/' + 'ids_universal_news_train_' + data_type + '.hdf5',
                      universal_hdf_dir+'ids_universal_2sets_train_70_30_' + data_type + '.hdf5',
                      int(configs["length_of_data"]*configs["train_part"]),
                      configs["percent_merge_2dsets"])
        merge_2_dsets(universal_hdf_dir + data_type + '_magazines/' + 'ids_universal_magazines_val_' + data_type +
                      '.hdf5',
                      universal_hdf_dir + data_type + '_news/' + 'ids_universal_news_val_' + data_type + '.hdf5',
                      universal_hdf_dir+'ids_universal_2sets_val_70_30_' + data_type + '.hdf5',
                      int(configs["length_of_data"]*configs["val_part"]),
                      configs["percent_merge_2dsets"])


if __name__ == "__main__":
    main()
