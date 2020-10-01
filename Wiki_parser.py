import shutil
from ForSasha import OrphoNet
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import json
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


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


def process(params, i):
    outputs = model.execute([params])
    df = pd.DataFrame(columns=['article_uuid', 'proc_sentence', 'proc_len', 'corrected'])
    if len(outputs) > 1 and outputs[0] == 'Incorrect':
        df.loc[i, 'corrected'] = outputs[1]
    return df


def list_multiprocessing(param_lst,
                         func,
                         **kwargs):
    workers = kwargs.pop('workers')

    with Pool(workers) as p:
        apply_lst = [([line, i], func, i, kwargs) for i, line in param_lst.iteritems() if line is not np.nan]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))

    # lists do not need such sorting, but this can be useful later
    result = sorted(result, key=lambda x: x[0])
    return [_[1] for _ in result]


def _apply_lst(args):

    params, func, num, kwargs = args
    return num, func(*params, **kwargs)


def get_answers_in_wiki(model):

    wikipedia = pd.read_csv("ruwiki_2018_09_25_test.csv", encoding='utf-8', sep=',')
    wikipedia['corrected'] = np.nan
    wikipedia['error_types'] = np.nan
    wikipedia['probability'] = np.nan
    # df = list_multiprocessing(wikipedia['proc_sentence'],
    #                           process,
    #                           workers=5)
    # df = pd.concat(df).reset_index()

    # outputs = [[i, model.execute(line)] for i, line in tqdm(enumerate(wikipedia['proc_sentence'].to_numpy()))
    #            if line is not np.nan]

    for i, line in tqdm(enumerate(wikipedia['proc_sentence'].to_numpy())):
        if line is not np.nan:
            output = model.execute(line)
            if len(output) > 1 and output[0] == 'Incorrect':
                wikipedia['corrected'].iloc[i] = output[1]
                wikipedia['error_types'].iloc[i] = output[3]
                wikipedia['probability'].iloc[i] = str(output[4])

    # for output in tqdm(outputs):
    #     i, answer = output
    #     if len(answer) > 1 and answer[0] == 'Incorrect':
    #         wikipedia['corrected'].iloc[i] = answer[1]
    wikipedia.dropna(subset=['article_uuid', 'proc_sentence', 'corrected'], how='any', inplace=True)
    wikipedia.to_csv("ruwiki_2018_09_25" + '_answered_fl_2_retrained' + '.csv')


def main(get_wiki=True, get_csv_fl_hardsoft_1=False):
    if get_wiki:
        model = OrphoNet()
        get_answers_in_wiki(model)
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
                data["weight_path"] = chkpth
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
    main()
