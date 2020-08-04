#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import re
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import time
from Class import TestPreprocess, ProcessOutput
from Model import TsyaModel
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
weight_path = "Chkpt_full_labels.pth"

    
def main(path_file):
    
    data_processor = TestPreprocess()
    
    if path_file:
        with open(path_file, 'r') as f:
            text_data = []
            for line in f:
                text_data.append(line.split('\n')[0])
    else:
        num_of_sentences = int(input("Число предложений: "))
        text_data = []
        for i in range(num_of_sentences):
            text = input("Предложение: ")
            text_data.append(text)

    start_time = time.time()
    data_with_tsya_or_nn = data_processor.check_contain_tsya_or_nn(text_data)
    if len(data_with_tsya_or_nn) == 0:
        message = ["Correct"]
    else:
        input_ids, mask_ids, prediction_dataloader, nopad = data_processor.process(text=data_with_tsya_or_nn)
        model = TsyaModel(weight_path = weight_path, train_from_chk = True)
        predicts = model.predict(prediction_dataloader, nopad)
        output = ProcessOutput()
        #incorrect, message, correct_text = output.process(predicts, input_ids, nopad, label_ids, text_data)
        all_messages, incorrect_words, correct_text_full, all_errors = output.process(predicts, input_ids, nopad, data_with_tsya_or_nn)
        #print(all_messages)
        #print(all_errors)
    print('Elapsed time: ', time.time() - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    my_args = parser.parse_args()
    path_to_file = my_args.file
    
    main(path_to_file)
