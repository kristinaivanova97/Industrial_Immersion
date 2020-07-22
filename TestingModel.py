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
from Class import TestPreprocess, TsyaModel, ProcessOutput
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
    
def main(path_file):
    
    data_processor = TestPreprocess()
    
    if path_file:
        with open(path_file, 'r') as f:
            text_data = []
            for line in f:
                text_data.append(line.split('\n')[0])
    else:
        # num_of_sentences = int(input("Число предложений: "))
        num_of_sentences = 1
        text_data = []
        for i in range(num_of_sentences):
            # text = input("Предложение: ")
            text = "Повесится можно"
            text_data.append(text)
        
    start_time = time.time()
    input_ids, mask_ids, prediction_dataloader, nopad, label_ids = data_processor.process(text=text_data)

    model = TsyaModel()
    
    # if len(text_data) == 1:
    #     predicts = model.predict_sentence(input_ids, mask_ids, nopad)
    #
    # else:
    predicts = model.predict_batch(prediction_dataloader, nopad)



    output = ProcessOutput()
    output.process(predicts, input_ids, nopad, label_ids, text_data)
    
    print('Elapsed time: ', time.time() - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    my_args = parser.parse_args()
    path_to_file = my_args.file
    
    main(path_to_file)
