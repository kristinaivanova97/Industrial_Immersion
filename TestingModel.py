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
from testing_class import TestPreprocess, TsyaModel
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def ProcessOutput(predictions, input_ids, nopad, data_tags, tokenizer):
    
    if len(predictions) < 2:
        
        toks = tokenizer.convert_ids_to_tokens(input_ids[0, :nopad[0]])
        text = tokenizer.decode(input_ids[0, :nopad[0]])
        fine_text = text.replace('[CLS] ', '').replace(' [SEP]', '')
        tags = np.array(data_tags[0][:nopad[0]])
        preds =  np.array(list(predictions[0]))
        
        coincide = np.sum(tags[(tags==4) | (tags==5)] == preds[(tags==4) | (tags==5)])
        print("Coincide in {} positions \n".format(coincide))
        
        if coincide == len(tags[(tags==4) | (tags==5)]):
            if (len(tags[(tags==4) | (tags==5)]) == 0):
                print("Sentence does not contain words with tsya/tsiya")
            else:
                print("Predicted and initial sentences coincide")
        else:
            print("Sentence contain a mistake!")

        print("Tokens = ", toks)
        print("Prediction = ", preds)
        print("Initial Tags = ", tags)
        print("Fine text = {} \n".format(fine_text))
        
    else:
        step = 0
        for i,predict in enumerate(predictions):
            for j, pred in enumerate(predict):
                toks = tokenizer.convert_ids_to_tokens(input_ids[step, :nopad[step]])
                text = tokenizer.decode(input_ids[step, :nopad[step]])
                fine_text = text.replace('[CLS] ', '').replace(' [SEP]', '')
                nomask_pred = pred[1:-1]
                tags =  np.array(data_tags[step][:nopad[step]])
                preds = np.array(pred)
                
                coincide = np.sum(tags[(tags==4) | (tags==5)] == preds[(tags==4) | (tags==5)])

                print("Coincide in {} positions with tsya/tsiya ".format(coincide))
                if coincide == len(tags[(tags==4) | (tags==5)]):
                    if (len(tags[(tags==4) | (tags==5)]) == 0):
                        print("Sentence does not contain words with tsya/tsiya")
                    else:
                        print("Predicted and initial sentences coincide")
                else:
                    print("Sentence contain a mistake!")

                print("Tokens = ", toks)
                print("Prediction = ", preds)
                print("Initial Tags = ", tags)

                print("Fine text = {} \n".format(fine_text))
                step+=1
    
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
    input_ids, mask_ids, prediction_dataloader, nopad, label_ids = data_processor.process(text=text_data)

    model = TsyaModel()
    
    if len(text_data) == 1:
        predicts = model.predict_sentence(input_ids, mask_ids, nopad)

    else:
        predicts = model.predict_batch(prediction_dataloader, nopad)
    
    ProcessOutput(predicts, input_ids, nopad, label_ids, tokenizer)
    
    print('Elapsed time: ', time.time() - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    my_args = parser.parse_args()
    path_to_file = my_args.file
    
    main(path_to_file)
