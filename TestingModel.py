import argparse
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from ForSasha import OrphoNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(path_file, write_from_terminal=False, nn_testing=True, tsya_testing=False, calculate_metrics_nn=True, calculate_metrics_tsya=False):

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
        # text_data = "Если находится внутри задачи, то написать что-то в форму обратной связи становится невозможно."
        # model = OrphoNet()
        # incorrect_words, all_messages, correct_text_full, all_errors, probs, probs_O = model.execute(["text_data"])
        # print(incorrect_words, all_messages, correct_text_full, all_errors, probs, probs_O)
    else:
        start_time = time.time()
        net = OrphoNet()

        if nn_testing:
            for test_file in ["test_nn/Gramota", "test_nn/Michail_collection"]:
                with open(test_file + '.txt', 'r') as file:
                    text = [line.strip() for line in file]

                with open(test_file + '_answered2.txt', 'w') as file:
                    for sentence in text:
                        sentence, _, _ = sentence.rpartition(',')
                        answer = net.execute([sentence])
                        if len(answer) > 1:
                            correct_sentence = answer[2]
                            message = answer[0][0]
                            if len(answer[3]) > 0:
                                mistake = answer[3]
                            else:
                                mistake = '-'
                        else:
                            correct_sentence = sentence
                            message = 'does not contain appropriate n/nn, tsya/tisya'
                            mistake = '-'
                        if message == 'Incorrect':
                            file.write('0' + ', ' + str(answer[4]) + ', ' + str(answer[5])+', ' + str(mistake) + ',' + correct_sentence + '\n')
                        elif message == 'does not contain appropriate n/nn, tsya/tisya':
                            file.write('2' + ', ' + str(mistake) + ',' + correct_sentence + '\n')
                        else:
                            file.write('1' + ', ' + str(mistake) + ',' + correct_sentence + '\n')
            with open('test_nn/Mozhno_itak_itak' + '.txt', 'r') as file:
                text = [line.strip() for line in file]

            with open('test_nn/Mozhno_itak_itak' + '_answered2.txt', 'w') as file:
                for sentence in text:
                    answer = net.execute([sentence])
                    if len(answer) > 1:
                        correct_sentence = answer[2]
                        message = answer[0][0]
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
            with open(test_file + '_answered2.txt', 'r') as file:
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
            with open(test_file + '_answered2.txt', 'r') as file:
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
            print("Full accuracy = {}".format(round(accuracy * 100, 2)))
            print("Michail acc = {}".format(round(acc1 * 100, 3)))
            print("Gramota acc = {}".format(round(acc2 * 100, 3)))

            y_true = np.concatenate((signs_true, signs_true2), axis=None)
            y_predict = np.concatenate((signs, signs2), axis=None)
            tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            print("Precision = {}, Recall = {}, F1 = {}".format(round(precision, 3), recall, round(f1, 3)))

        if tsya_testing:
            data = pd.read_csv('/mnt/sda/orpho/data/test_linguists.csv', index_col=0)
            data['new_model_v2'] = np.nan
            data['new_model_correct'] = np.nan
            for i in range(len(data)):
                sentence = data['text'].iloc[i]
                answer = net.execute([sentence])
                correct_sentence = answer[2]
                data['new_model_v2'].iloc[i] = '+' if answer[0][0][0] == 'Correct' else '-'
                data['new_model_correct'].iloc[i] = correct_sentence[0]
            data.to_csv('test_linguists_answered2.csv')

        if calculate_metrics_tsya:
            data = pd.read_csv('/mnt/sda/orpho/data/test_linguists_answered2.csv', index_col=0)
            data_true = pd.read_csv('/mnt/sda/orpho/data/test_linguists.csv', index_col=0)
            accuracy = np.sum(np.where(np.asarray(data['new_model_v2']) == np.asarray(data['y_true']), 1, 0))/len(data_true['y_true'])
            print("Tsya accuracy = {}".format(round(accuracy*100, 3)))
        print('Elapsed time: ', time.time() - start_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    my_args = parser.parse_args()
    path_to_file = my_args.file
    
    main(path_to_file, write_from_terminal=False, nn_testing=True, tsya_testing=False, calculate_metrics_nn=True, calculate_metrics_tsya=False)
