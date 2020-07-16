#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import re
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertConfig
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time
from testing_class import TestPreprocess, TsyaModel
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

'''
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat[labels_flat!=0] == labels_flat[labels_flat!=0]) / len(labels_flat[labels_flat!=0])
'''
def main(path_file):
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
    
    data_processor = TestPreprocess()
    
    if path_to_file:
        f = open(path_to_file, 'r')
        text_data = []
        for line in f:
            text_data.append(line.split('\n')[0])
        f.close()
    else:
        num_of_sentences = int(input("Число предложений: "))
        text_data = []
        for i in range(num_of_sentences):
            text = input("Предложение: ")
            text_data.append(text)
        
    start_time = time.time()
    input_ids, mask_ids, prediction_dataloader, nopad = data_processor.process(text=text_data)
    model = TsyaModel()
    if len(text_data) == 1:
        predicts = model.predict_sentence(input_ids, mask_ids, nopad)
        print(tokenizer.convert_ids_to_tokens(input_ids[0, :nopad[0]]))
        print(predicts)

    else:
        predicts = model.predict_batch(prediction_dataloader, nopad)
        step = 0
        for i,predict in enumerate(predicts):
            for j, pred in enumerate(predict):
                toks = tokenizer.convert_ids_to_tokens(input_ids[step, :nopad[step]])
                print(toks)
                print(pred)
                step+=1
    print('Elapsed time: ', time.time() - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    my_args = parser.parse_args()
    path_to_file = my_args.file
    
    main(path_to_file)
