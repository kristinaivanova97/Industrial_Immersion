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
    input_ids, mask_ids, prediction_dataloader, nopad, label_ids = data_processor.process(text=text_data)

    model = TsyaModel(weight_path = weight_path, train_from_chk = True)
    
    #if len(text_data) == 1:
    #    predicts = model.predict_sentence(input_ids, mask_ids, nopad)
    
    predicts = model.predict_batch(prediction_dataloader, nopad)
    output = ProcessOutput()
    #incorrect, message, correct_text = output.process(predicts, input_ids, nopad, label_ids, text_data)
    incorrect, message, correct_text = output.process(predicts, input_ids, nopad, text_data)
    
    print('Elapsed time: ', time.time() - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    my_args = parser.parse_args()
    path_to_file = my_args.file
    
    main(path_to_file)
