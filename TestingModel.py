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
weight_path = "Chkpt2.pth"

def check_contain_tsya_or_nn(data):
    
    data_with_tsya_or_nn = []
    tsya_search = re.compile(r'тся\b')
    tsiya_search = re.compile(r'ться\b')
    nn_search = re.compile(r'\wнн([аоы]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых)\b', re.IGNORECASE) # the words, which contain "н" in the middle or in the end of word
    n_search = re.compile(r'[аоэеиыуёюя]н([аоы]|ый|ого|ому|ом|ым|ая|ой|ую|ые|ыми|ых)\b', re.IGNORECASE)
    
    for sentence in data:
        
        places_with_tsya = tsya_search.search(sentence)
        places_with_tisya = tsiya_search.search(sentence)
        places_with_n = n_search.search(sentence)
        places_with_nn = nn_search.search(sentence)
        
        if (places_with_tsya is not None) or (places_with_tisya is not None) or (places_with_n is not None) or (places_with_nn is not None):
            data_with_tsya_or_nn.append(sentence)
        
    return data_with_tsya_or_nn
    
def main(path_file):
    
    data_processor = TestPreprocess()
    
    if path_file:
        with open(path_file, 'r') as f:
            text_data = []
            for line in f:
                text_data.append(line.split('\n')[0])
    else:
        num_of_sentences = int(input("Число предложений: "))
        # num_of_sentences = 1
        text_data = []
        for i in range(num_of_sentences):
            text = input("Предложение: ")
            # text = 'Мне плохо спиться'
            text_data.append(text)

    start_time = time.time()
    data_with_tsya_or_nn = check_contain_tsya_or_nn(text_data)
    if len(data_with_tsya_or_nn) == 0:
        message = ["Correct"]
    else:
        input_ids, mask_ids, prediction_dataloader, nopad = data_processor.process(text=data_with_tsya_or_nn)
        model = TsyaModel(weight_path = weight_path, train_from_chk = True)
        predicts = model.predict(prediction_dataloader, nopad)
        output = ProcessOutput()
        incorrect_for_sentences, all_messages, correct_text_full, all_errors = output.process(predicts, input_ids, nopad, data_with_tsya_or_nn)

    print('Elapsed time: ', time.time() - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    my_args = parser.parse_args()
    path_to_file = my_args.file
    
    main(path_to_file)
