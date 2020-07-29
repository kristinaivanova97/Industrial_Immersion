import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import random
from tqdm import tqdm
from Model import GetIndices, TsyaModel

batch_size = 1
epochs = 3 # The BERT authors recommend between 2 and 4.
max_seq_length = 512 # for bert this limit exists
data_dir = "./new_data/"
chkp_path = "Chkpt_part_of_word.pth"

def main():

    train_data_processor = GetIndices(ftype = 'train', data_dir='./data/')
    train_data_processor.upload()
    assert len(train_data_processor.input_ids[0]) == max_seq_length
    assert len(train_data_processor.input_mask[0]) == max_seq_length
    assert len(train_data_processor.label_ids[0]) == max_seq_length

    print("Sequense train len = ", len(train_data_processor.input_ids[0]))
    print("Num of sequences = ", len(train_data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    val_data_processor = GetIndices(ftype='val', data_dir='./data/')
    val_data_processor.upload()
    assert len(val_data_processor.input_ids[0]) == max_seq_length
    assert len(val_data_processor.input_mask[0]) == max_seq_length
    assert len(val_data_processor.label_ids[0]) == max_seq_length

    print("Sequense val len = ", len(val_data_processor.input_ids[0]))
    print("Num of sequences = ", len(val_data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    #model = TsyaModel(TrainProcessor = TrainProcessor, ValProcessor = ValProcessor)
    model = TsyaModel(weight_path=chkp_path, train_from_chk=True)
    model.train(chkp_path, train_data_processor, val_data_processor)


if __name__ == "__main__":
    main()

