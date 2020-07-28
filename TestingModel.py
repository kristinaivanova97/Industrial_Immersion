#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import re
import torch
import time
from testing_class import TestPreprocess, TsyaModel, ProcessOutput
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
weight_path = "Chkpt2.pth"
    
def check_contain_tsya_or_nn(data):
    
    data_with_tsya_or_nn = []
    tsya_search = re.compile(r'тся\b')
    tsiya_search = re.compile(r'ться\b')
    n_search = re.compile(r'\wн*\w\b')
    nn_search = re.compile(r'\wнн*\w\b')
    
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
    data_with_tsya_or_nn = check_contain_tsya_or_nn(text_data)
    if len(data_with_tsya_or_nn) == 0:
        message = ["Correct"]
    else:
        input_ids, mask_ids, prediction_dataloader, nopad, label_ids = data_processor.process(text=data_with_tsya_or_nn)
        model = TsyaModel(weight_path = weight_path, train_from_chk = True)
        predicts = model.predict_batch(prediction_dataloader, nopad)
        output = ProcessOutput()
        #incorrect, message, correct_text = output.process(predicts, input_ids, nopad, label_ids, text_data)
        incorrect, message, correct_text = output.process(predicts, input_ids, nopad, data_with_tsya_or_nn)
    
    print('Elapsed time: ', time.time() - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    my_args = parser.parse_args()
    path_to_file = my_args.file
    
    main(path_to_file)
