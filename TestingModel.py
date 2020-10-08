from pathlib import Path
from sklearn.metrics import confusion_matrix
from Class_for_execution import OrphoNet
import numpy as np
import pandas as pd
import argparse
import time
import json
import csv
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def extract_signs(test_file, suffix):
    with open(test_file + suffix + '.txt', 'r') as file:
        text = [line.strip() for line in file]
        signs = np.empty(len(text))
        for i, sentence in enumerate(text):
            sign, sentence = sentence.split(',', 1)
            signs[i] = int(sign)

    with open(test_file + '.txt', 'r') as file:
        text = [line.strip() for line in file]
        signs_true = np.empty(len(text))
        for i, sentence in enumerate(text):
            sentence, _, sign = sentence.rpartition(',')
            if sign == '-':
                signs_true[i] = 0
            else:
                signs_true[i] = 1
    return signs_true, signs


def print_to_file(net, file, sentence, default_value):

    answer = net.execute(sentence, default_value)
    if len(answer) > 1:
        correct_sentence = answer[1]
        message = answer[0]
        if len(answer[3]) > 0:
            mistake = answer[3]
        else:
            mistake = '-'
    else:
        correct_sentence = sentence
        message = 'does not contain appropriate n/nn, tsya/tisya'
        mistake = '-'
    if message == 'Incorrect':
        file.write('0' + ', ' + str(answer[4]) + ', ' + str(answer[5]) + ', ' + str(mistake) + ',' +
                   correct_sentence + '\n')
    elif message == 'does not contain appropriate n/nn, tsya/tisya':
        file.write('2' + ', ' + str(mistake) + ',' + correct_sentence + '\n')
    else:
        file.write('1' + ', ' + str(mistake) + ',' + correct_sentence + '\n')


def compute_metrics(y_true, y_predict, test_file):
    acc = np.sum(np.where(np.asarray(y_true) == np.asarray(y_predict), 1, 0)) / len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
    precision = tn / (tn + fn)
    recall = tn / (tn + fp)
    f1 = 2 * precision * recall / (precision + recall)
    print(test_file + " Precision = {}, Recall = {}, F1 = {}".format(round(precision, 4), round(recall, 4),
                                                                     round(f1, 4)))
    return acc, precision, recall, f1


def test(writer, model_name, net, suffix, nn_testing, tsya_testing, calculate_metrics_nn, calculate_metrics_tsya,
         default_value, test_nn_dir):

    acc1, precision1, recall1, f11 = np.nan, np.nan, np.nan, np.nan
    acc2, precision2, recall2, f12 = np.nan, np.nan, np.nan, np.nan
    acc, precision, recall, f1 = np.nan, np.nan, np.nan, np.nan
    acc_tsya, precision_tsya, recall_tsya, f1_tsya = np.nan, np.nan, np.nan, np.nan

    if nn_testing:
        for test_file in [test_nn_dir + "Gramota_doubled", test_nn_dir + "Michail_collection_doubled"]:
            with open(test_file + '.txt', 'r') as file:
                text = [line.strip() for line in file]

            with open(test_file + suffix + '.txt', 'w') as file:
                for sentence in text:
                    sentence, _, _ = sentence.rpartition(',')
                    print_to_file(net, file, sentence, default_value)
        with open(test_nn_dir + 'Mozhno_itak_itak_doubled' + '.txt', 'r') as file:
            text = [line.strip() for line in file]

        with open(test_nn_dir + 'Mozhno_itak_itak_doubled' + suffix + '.txt', 'w') as file:
            for sentence in text:
                print_to_file(net, file, sentence, default_value)

    if calculate_metrics_nn:
        test_file1 = test_nn_dir + "Michail_collection_doubled"
        signs_true, signs = extract_signs(test_file1, suffix)
        test_file2 = test_nn_dir + "Gramota_doubled"
        signs_true2, signs2 = extract_signs(test_file2, suffix)

        signs_true = signs_true[signs != 2]
        signs_true2 = signs_true2[signs2 != 2]
        signs = signs[signs != 2]
        signs2 = signs2[signs2 != 2]

        acc1, precision1, recall1, f11 = compute_metrics(signs_true, signs, "Michail")
        acc2, precision2, recall2, f12 = compute_metrics(signs_true2, signs2, "Gramota")
        _, precision, recall, f1 = compute_metrics(np.concatenate((signs_true, signs_true2), axis=None),
                                                   np.concatenate((signs, signs2), axis=None), "Full")
        acc = (acc1 + acc2) / 2
        print("Full n/nn accuracy = {}".format(round(acc * 100, 2)))
        print("Michail acc = {}".format(round(acc1 * 100, 3)))
        print("Gramota acc = {}".format(round(acc2 * 100, 3)))

    if tsya_testing:
        data = pd.read_csv('/mnt/sda/orpho/data/test_linguists.csv', index_col=0)
        data['new_model_v2'] = np.nan
        data['new_model_correct'] = np.nan
        for i in range(len(data)):
            sentence = data['text'].iloc[i]
            answer = net.execute(sentence, default_value)
            if len(answer) > 1:
                correct_sentence = answer[1]
                data['new_model_v2'].iloc[i] = '-'
                data['new_model_correct'].iloc[i] = correct_sentence
            else:
                data['new_model_v2'].iloc[i] = '+'
                data['new_model_correct'].iloc[i] = sentence

        data.to_csv('./test_linguists/' + 'test_linguists' + suffix + '.csv')

    if calculate_metrics_tsya:
        data = pd.read_csv('./test_linguists/' + 'test_linguists' + suffix + '.csv', index_col=0)
        data_true = pd.read_csv('/mnt/sda/orpho/data/test_linguists.csv', index_col=0)
        _, precision_tsya, recall_tsya, f1_tsya = compute_metrics(data_true['y_true'].to_numpy(),
                                                                  data['new_model_v2'].to_numpy(), "Tsya")
        acc_tsya = np.sum(np.where(np.asarray(data['new_model_v2']) == np.asarray(data['y_true']), 1, 0)) / len(
            data_true['y_true'])
        print("Tsya accuracy = {}".format(round(acc_tsya * 100, 3)))

    writer.writerow([model_name, acc1, precision1, recall1, f11, acc2, precision2, recall2, f12,
                    acc, precision, recall, f1, acc_tsya, precision_tsya, recall_tsya, f1_tsya])


def main(path_file, configs):
    write_from_terminal = configs["write_from_terminal"]
    nn_testing = configs["nn_testing"]
    tsya_testing = configs["tsya_testing"]
    calculate_metrics_nn = configs["calculate_metrics_nn"]
    calculate_metrics_tsya = configs["calculate_metrics_tsya"]
    default_value = ["default_value"]

    if path_file:
        with open(path_file, 'r') as f:
            text_data = []
            for line in f:
                text_data.append(line.split('\n')[0])
    elif write_from_terminal:
        num_of_sentences = int(input("Число предложений: "))
        text_data = []
        for i in range(num_of_sentences):
            text = input("Предложение: ")
            text_data.append(text)

    else:
        start_time = time.time()
        list_of_models = configs["list_of_models"]

        suffixes = ["_answered_" + model for model in list_of_models]
        chkpths = ["Chkpt_" + model + ".pth" for model in list_of_models]
        # suffixes = ['_answered2', '_answered_fl_hs_1set', '_answered_fl_hs_2set', '_answered_fl_hs_schit_1set',
        #             '_answered_fl_hs_schit_2set', '_answered', '_answered_full_endings_1set',
        #             '_answered_full_endings_2set', '_answered_more_nn_sent_1set', '_answered_full_end_more_nn_2set']
        # chkpths = ['Chkpt_full_labels.pth', 'Chkpt_fl_hardsoft_1set.pth', 'Chkpt_fl_hardsoft_2set.pth',
        #            'Chkpt_fl_hardsoft_schitanye_1set.pth', 'Chkpt_fl_hardsoft_schitanye_2set.pth',
        #            'Chkpt_part_of_word.pth', 'Chkpt_pow_new_endings_1set.pth', 'Chkpt_pow_new_endings_2set.pth',
        #            'Chkpt_pow_new_endings_1set_test.pth', "Chkpt_pow_new_endings_2set_test.pth"]
        
        # Path(configs[configs["dir_comparison_file"]]).mkdir(parents=True, exist_ok=True)
        with open(configs["dir_comparison_file"]+configs["comparison_file"], 'w', newline='') as csvFile:

            writer = csv.writer(csvFile)
            writer.writerow(["Model", "Mihail_acc", "Mihail_p", "Mihail_r", "Mihail_f1",
                             "Gramota_acc", "Gramota_p", "Gramota_r", "Gramota_f1",
                             "Full_acc", "Full_p", "Full_r", "Full_f1",
                             "Tsya_acc", "Tsya_p", "Tsya_r", "Tsya_f1"])

            # ['FL_hardsoft_1', 'FL_hardsoft_2', 'FL_hardsoft_2_70_30',
            #  'POW_hardsoft_1', 'POW_hardsoft_2']
            # ['FL_hard_1', 'FL_hardsoft_1', 'FL_hardsoft_2', 'FL_hardsoft_1_schitanye',
            #  'FL_hardsoft_2_schitanye', 'POW_hard_1', 'POW_hardsoft_1', 'POW_hardsoft_2',
            #  'POW_hardsoft_1_schitanye', 'POW_hardsoft_2_schitanye']
            for i, model_name in enumerate(list_of_models):
                with open("config_stand.json", "r+") as jsonFile:
                    data = json.load(jsonFile)
                    data["chckp_file"] = chkpths[i]
                    print(model_name)
                    if i != 6:
                        data["from_rubert"] = False
                        if i > 3:
                            data['label_list'] = ["[PAD]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n",
                                                  "REPLACE_tysya", "REPLACE_tsya", "[##]"]
                        else:
                            data['label_list'] = ["[PAD]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n",
                                                  "REPLACE_tysya", "REPLACE_tsya"]
                    else:
                        data["from_rubert"] = True
                        data['label_list'] = ["[PAD]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n",
                                              "REPLACE_tysya", "REPLACE_tsya"]

                    jsonFile.seek(0)  # rewind
                    json.dump(data, jsonFile)
                    jsonFile.truncate()
                net = OrphoNet()
                test(writer, model_name, net, suffixes[i], nn_testing, tsya_testing, calculate_metrics_nn,
                     calculate_metrics_tsya, default_value, test_nn_dir="./test_nn/")
                print('Elapsed time: ', time.time() - start_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    my_args = parser.parse_args()
    path_to_file = my_args.file
    with open("config_stand.json") as json_data_file:
        configs = json.load(json_data_file)

    main(path_to_file, configs=configs)
