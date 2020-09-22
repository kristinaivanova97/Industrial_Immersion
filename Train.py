import json

import torch
from transformers import BertTokenizer, AutoTokenizer

from Model import GetIndices, TsyaModel
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def main():

    with open("config_train.json") as json_data_file:
        configs = json.load(json_data_file)

    train_data_processor = GetIndices(ftype=configs["train_ftype"], data_dir=configs["data_path"])
    train_data_processor.upload_hdf()
    assert len(train_data_processor.input_ids[0]) == configs["max_seq_length"]
    assert len(train_data_processor.input_mask[0]) == configs["max_seq_length"]
    assert len(train_data_processor.label_ids[0]) == configs["max_seq_length"]

    print("Sequense train len = ", len(train_data_processor.input_ids[0]))
    print("Num of sequences = ", len(train_data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    val_data_processor = GetIndices(ftype=configs["val_ftype"], data_dir=configs["data_path"])
    val_data_processor.upload_hdf()
    assert len(val_data_processor.input_ids[0]) == configs["max_seq_length"]
    assert len(val_data_processor.input_mask[0]) == configs["max_seq_length"]
    assert len(val_data_processor.label_ids[0]) == configs["max_seq_length"]

    print("Sequense val len = ", len(val_data_processor.input_ids[0]))
    print("Num of sequences = ", len(val_data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    if configs['from_rubert']:
        tokenizer = BertTokenizer.from_pretrained(**configs['config_of_tokenizer'])

    else:
        tokenizer = AutoTokenizer.from_pretrained(**configs['config_of_tokenizer'])

    adam_options = configs['adam_options']
    model = TsyaModel(label_list=configs["label_list"], weight_path=None, train_from_chk=configs["train_from_chk"])
    model.train(train_data_processor, val_data_processor, chkp_path=configs["weight_path"], epochs=configs["epochs"],
                batch_size=configs["batch_size"])


if __name__ == "__main__":
    main()