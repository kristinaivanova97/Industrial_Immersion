from sklearn.metrics import confusion_matrix
from ForSasha import OrphoNet
import numpy as np
import pandas as pd
import argparse
import time
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def test(model_name, net, suffix, nn_testing, tsya_testing, calculate_metrics_nn, calculate_metrics_tsya, default_value):
    if nn_testing:
        for test_file in ["test_nn/Gramota", "test_nn/Michail_collection"]:
            with open(test_file + '.txt', 'r') as file:
                text = [line.strip() for line in file]

            with open(test_file + suffix + '.txt', 'w') as file:
                for sentence in text:
                    sentence, _, _ = sentence.rpartition(',')
                    answer = net.execute_old([sentence], default_value)
                    if len(answer) > 1:
                        correct_sentence = answer[2]
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
                        file.write('0' + ', ' + str(answer[4]) + ', ' + str(answer[5]) + ', ' + str(
                            mistake) + ',' + correct_sentence + '\n')
                    elif message == 'does not contain appropriate n/nn, tsya/tisya':
                        file.write('2' + ', ' + str(mistake) + ',' + correct_sentence + '\n')
                    else:
                        file.write('1' + ', ' + str(mistake) + ',' + correct_sentence + '\n')
        with open('test_nn/Mozhno_itak_itak' + '.txt', 'r') as file:
            text = [line.strip() for line in file]

        with open('test_nn/Mozhno_itak_itak' + suffix + '.txt', 'w') as file:
            for sentence in text:
                answer = net.execute_old([sentence], default_value)
                if len(answer) > 1:
                    correct_sentence = answer[2]
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

    if calculate_metrics_nn:
        test_file = "test_nn/Michail_collection"
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

        test_file = "test_nn/Gramota"
        with open(test_file + suffix + '.txt', 'r') as file:
            text = [line.strip() for line in file]
            signs2 = np.empty(len(text))
            for i, sentence in enumerate(text):
                sign, sentence = sentence.split(',', 1)
                signs2[i] = int(sign)

        with open(test_file + '.txt', 'r') as file:
            text = [line.strip() for line in file]
            signs_true2 = np.empty(len(text))
            for i, sentence in enumerate(text):
                sentence, _, sign = sentence.rpartition(',')
                if sign == '-':
                    signs_true2[i] = 0
                else:
                    signs_true2[i] = 1
        signs_true = signs_true[signs != 2]
        signs_true2 = signs_true2[signs2 != 2]
        signs = signs[signs != 2]
        signs2 = signs2[signs2 != 2]
        acc1 = np.sum(np.where(np.asarray(signs_true) == np.asarray(signs), 1, 0)) / len(signs_true)
        acc2 = np.sum(np.where(np.asarray(signs_true2) == np.asarray(signs2), 1, 0)) / len(signs_true2)
        accuracy = (acc1 + acc2) / 2
        print("Full n/nn accuracy = {}".format(round(accuracy * 100, 2)))
        print("Michail acc = {}".format(round(acc1 * 100, 3)))
        print("Gramota acc = {}".format(round(acc2 * 100, 3)))

        y_true = np.concatenate((signs_true, signs_true2), axis=None)
        y_predict = np.concatenate((signs, signs2), axis=None)
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        precision = tn / (tn + fn)
        recall = tn / (tn + fp)
        f1 = 2 * precision * recall / (precision + recall)
        print("Precision = {}, Recall = {}, F1 = {}".format(round(precision, 3), round(recall, 3), round(f1, 3)))

    if tsya_testing:
        data = pd.read_csv('/mnt/sda/orpho/data/test_linguists.csv', index_col=0)
        data['new_model_v2'] = np.nan
        data['new_model_correct'] = np.nan
        for i in range(len(data)):
            sentence = data['text'].iloc[i]
            answer = net.execute_old([sentence], default_value)
            correct_sentence = answer[2]
            data['new_model_v2'].iloc[i] = '+' if answer[0] == 'Correct' else '-'
            data['new_model_correct'].iloc[i] = correct_sentence
        data.to_csv('test_linguists' + suffix + '.csv')

    if calculate_metrics_tsya:
        data = pd.read_csv('test_linguists' + suffix + '.csv', index_col=0)
        data_true = pd.read_csv('/mnt/sda/orpho/data/test_linguists.csv', index_col=0)
        acc = np.sum(np.where(np.asarray(data['new_model_v2']) == np.asarray(data['y_true']), 1, 0)) / len(
            data_true['y_true'])
        tn, fp, fn, tp = confusion_matrix(data_true['y_true'].to_numpy(), data['new_model_v2'].to_numpy()).ravel()
        precision = tn / (tn + fn)
        recall = tn / (tn + fp)
        f1 = 2 * precision * recall / (precision + recall)
        print("Tsya accuracy = {}".format(round(acc * 100, 3)))
        print("Precision = {}, Recall = {}, F1 = {}".format(round(precision, 3), round(recall, 3), round(f1, 3)))


def main(path_file, write_from_terminal, nn_testing, tsya_testing, calculate_metrics_nn, calculate_metrics_tsya, default_value):

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
        net = OrphoNet()
        suffixes = ['_answered', '_answered2', '_full_endings_1set', '_answered_more_nn_sent_1set',
                                                                        '_answered_full_end_more_nn_2set']
        chkpths = ['Chkpt_part_of_word.pth', 'Chkpt_full_labels.pth', 'Chkpt_pow_new_endings_1set.pth',
                    'Chkpt_pow_new_endings_1set_test.pth', "Chkpt_pow_new_endings_2set_test.pth"]

        for i, model_name in enumerate(['POW_hard_1', 'FL_hard_1', 'POW_hardsoft_1', 'POW_hardsoft_1_schitanye',
                                                                                'POW_hardsoft_2_schitanye']):
            with open("config_stand.json", "r+") as jsonFile:
                data = json.load(jsonFile)
                data["weight_path"] = chkpths[i]
                jsonFile.seek(0)  # rewind
                json.dump(data, jsonFile)
                jsonFile.truncate()

            test(model_name, net, suffixes[i], nn_testing, tsya_testing, calculate_metrics_nn, calculate_metrics_tsya, default_value)
            print('Elapsed time: ', time.time() - start_time)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    my_args = parser.parse_args()
    path_to_file = my_args.file
    
    main(path_to_file, write_from_terminal=False, nn_testing=True, tsya_testing=True, calculate_metrics_nn=True, calculate_metrics_tsya=True, default_value='Incorrect')
