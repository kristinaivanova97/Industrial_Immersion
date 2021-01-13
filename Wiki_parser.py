import shutil
from Class_for_execution import OrphoNet
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import json
import csv
import re
import parmap
# import multiprocessing
# from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# model = OrphoNet()

def choose_texts():
    # цикл по всем папкам и добавление их данных в список
    root = "./AA/"
    out_dir = './wiki_answered/'
    # for file in tqdm(os.listdir(root)):
    #    if file == 'wiki_02':
    file = 'wiki_00'
    with open(out_dir + file + "_answered.txt", 'w', encoding="utf-8") as output_file:
        print(file)
        with open(root + file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines[3:-6]]
        outputs = [model.execute([line]) for line in lines]
        for answer in tqdm(outputs):
            if len(answer) > 1 and answer[0] == 'Incorrect':
                output_file.writelines(str(answer[1]) + '\n')
    root = '../text/'
    dirs = [src.replace('../text/', '') for src, _, _ in os.walk(root)]
    print(dirs)
    for d in dirs:
        if d != '':
            print(d)
            shutil.copy(root + d + '/wiki_00', "different_texts_wiki/wiki_00_" + d)
            shutil.copy(root + d + '/wiki_01', "different_texts_wiki/wiki_01_" + d)


def process(data, model):
    for i, line in tqdm(enumerate(data['proc_sentence'].to_numpy())):
        if line is not np.nan:
            output = model.execute(line)
            if len(output) > 1 and output[0] == 'Incorrect':

                data.loc[i, 'corrected'] = output[1]
                data.loc[i, 'error_types'] = str(output[3])
                data.loc[i, 'probability'] = str(output[4])
    return data


def list_multiprocessing(data, model):
    num_processes = 2
    chunk_size = int(data.shape[0] / num_processes)

    chunks = [data.iloc[i:i + chunk_size] for i in range(0, data.shape[0], chunk_size)]
    # pool = multiprocessing.Pool(processes=num_processes)
    # multiprocessing.freeze_support()
    # result = pool.map(process, chunks)
    # pool.close()
    result = parmap.map(process, chunks, model, pm_chunksize=chunk_size, pm_processes=num_processes, pm_pbar=True)

    for i in tqdm(range(len(result))):
        data.iloc[result[i].index] = result[i]
    data = data.dropna(subset=['article_uuid', 'proc_sentence', 'corrected'], how='any', inplace=True)
    return data


def get_answers_in_wiki(model):

    f = open("ruwiki_full.csv", encoding='utf-8')
    csv_reader = csv.reader(f)
    with open("ruwiki_full" + '_answered_fl_doubled_sent_500k_2020_12_17' + '.csv', 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['index', 'sentence', 'corrected', 'error_types', 'probability'])
    # print(next(csv_reader))
    # wikipedia = pd.read_csv("ruwiki_full.csv", encoding='utf-8', sep=',')
    # wikipedia['corrected'] = np.nan
    # wikipedia['error_types'] = np.nan
    # wikipedia['probability'] = np.nan
    # df = list_multiprocessing(wikipedia['proc_sentence'],
    #                           process,
    #                           workers=5)
    # df = pd.concat(df).reset_index()

    # outputs = [[i, model.execute(line)] for i, line in tqdm(enumerate(wikipedia['proc_sentence'].to_numpy()))
    #            if line is not np.nan]
    header = next(csv_reader)
    i = -1
    data = []
    for line in tqdm(csv_reader):
        if line[2] == line[2]:
            i += 1
            output = model.execute(line[2])
            if len(output) > 1 and output[0] == 'Incorrect':
                data.append([line[0], line[2], output[1], str(output[3]), str(output[4])])
            if i % 1000 == 0:
                with open("ruwiki_full" + '_answered_fl_doubled_sent_500k_2020_12_17' + '.csv', 'a+', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(data)
                data = []

    # for i, line in tqdm(enumerate(wikipedia['proc_sentence'].to_numpy())):
    #     if line is not np.nan:
    #         output = model.execute(line)
    #         if len(output) > 1 and output[0] == 'Incorrect':
    #             # brace_open = re.compile(r'{')
    #             # brace_close = re.compile(r'}')
    #             # brace_open_escape = re.escape('{')
    #             # brace_close_escape = re.escape('}')
    #             # output[1] = output[1].replace('{', '\\{').replace('}', '\\}')
    #             # correct = re.compile(rf'{output[1]}')
    #
    #             wikipedia.loc[i, 'corrected'] = output[1]
    #             wikipedia.loc[i, 'error_types'] = str(output[3])
    #             wikipedia.loc[i, 'probability'] = str(output[4])

    # wikipedia.dropna(subset=['article_uuid', 'proc_sentence', 'corrected'], how='any', inplace=True)
    # wikipedia.to_csv("ruwiki_full" + '_answered_fl_doubled_sent_500k_2020_12_17' + '.csv', index=False)


def main(get_wiki=True, get_csv_fl_hardsoft_1=False):
    if get_wiki:
        model = OrphoNet()
        get_answers_in_wiki(model)

        # *** parallel part ***
        # wikipedia = pd.read_csv("ruwiki_full.csv", encoding='utf-8', sep=',')
        # wikipedia['corrected'] = np.nan
        # wikipedia['error_types'] = np.nan
        # wikipedia['probability'] = np.nan
        # wikipedia = list_multiprocessing(wikipedia, model)
        # wikipedia.to_csv("ruwiki_full" + '_answered_fl_doubled_sent_500k_2020_12_17' + '.csv', index=False)
        # *** end ***
    if get_csv_fl_hardsoft_1:
        start_time = time.time()
        models = []
        # TODO retrain model on old dataset for FL_hardsoft_1
        chkpths = ['Chkpt_fl_hardsoft_correct_1set.pth', 'Chkpt_fl_hardsoft_correct_1set_old.pth']
        out_df = pd.DataFrame(columns=['original_sent', 'fl_hs_1_universal', 'fl_hs_1_old'])
        both_texts = []
        for test_file in ["test_nn/Gramota_doubled", "test_nn/Michail_collection_doubled"]:
            with open(test_file + '.txt', 'r') as file:
                text = [line.strip() for line in file]
            both_texts += text
        for i, chkpth in enumerate(chkpths):
            with open("config_stand.json", "r+") as jsonFile:
                data = json.load(jsonFile)
                data["chckp_file"] = chkpth
                jsonFile.seek(0)
                json.dump(data, jsonFile)
                jsonFile.truncate()
            models.append(OrphoNet())
        for j, sentence in enumerate(both_texts):
            sentence, _, _ = sentence.rpartition(',')
            out_df.loc[j, 'original_sent'] = sentence
            output = models[0].execute(sentence)
            output2 = models[1].execute(sentence)
            if len(output) > 1 and output[0] == 'Incorrect':
                out_df['fl_hs_1_universal'].iloc[j] = output[1]
            else:
                out_df['fl_hs_1_universal'].iloc[j] = "Correct"
            if len(output2) > 1 and output2[0] == 'Incorrect':
                out_df['fl_hs_1_old'].iloc[j] = output2[1]
            else:
                out_df['fl_hs_1_old'].iloc[j] = "Correct"
        out_df.to_csv('compare_fl_hardsoft_1' + '.csv')

        print('Elapsed time: ', time.time() - start_time)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')

    # model = OrphoNet()
    # wikipedia = pd.read_csv("ruwiki_full.csv", encoding='utf-8', sep=',')
    # wikipedia['corrected'] = np.nan
    # wikipedia['error_types'] = np.nan
    # wikipedia['probability'] = np.nan
    # wikipedia = list_multiprocessing(wikipedia, model)
    # wikipedia.to_csv("ruwiki_full" + '_answered_fl_doubled_sent_500k_2020_12_17' + '.csv', index=False)
    main()
