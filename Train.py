import json

import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from Model import GetIndices, TsyaModel

max_seq_length = 512 # for bert this limit exists
#chkp_path = "Chkpt_part_of_word_new_deeppavlov.pth"

def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    train_data_processor = GetIndices(ftype='train', data_dir=config['output_dir_train_val_test_split'])
    train_data_processor.upload_hdf()
    assert len(train_data_processor.input_ids[0]) == max_seq_length
    assert len(train_data_processor.input_mask[0]) == max_seq_length
    assert len(train_data_processor.label_ids[0]) == max_seq_length

    print("Sequense train len = ", len(train_data_processor.input_ids[0]))
    print("Num of sequences = ", len(train_data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    val_data_processor = GetIndices(ftype='val', data_dir=config['output_dir_train_val_test_split'])
    val_data_processor.upload_hdf()
    assert len(val_data_processor.input_ids[0]) == max_seq_length
    assert len(val_data_processor.input_mask[0]) == max_seq_length
    assert len(val_data_processor.label_ids[0]) == max_seq_length

    print("Sequense val len = ", len(val_data_processor.input_ids[0]))
    print("Num of sequences = ", len(val_data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    model = TsyaModel(weight_path=None, train_from_chk=False)
    model.train(config['chkp_path'], train_data_processor, val_data_processor)


if __name__ == "__main__":
    main()
