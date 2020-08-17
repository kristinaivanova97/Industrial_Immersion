import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from Model import GetIndices, TsyaModel

batch_size = 6
epochs = 3 # The BERT authors recommend between 2 and 4.
max_seq_length = 512 # for bert this limit exists
chkp_path = "Chkpt_part_of_word_new.pth"

def main():

    train_data_processor = GetIndices(ftype='train', data_dir='./new_data_split/')
    train_data_processor.upload_hdf()
    assert len(train_data_processor.input_ids[0]) == max_seq_length
    assert len(train_data_processor.input_mask[0]) == max_seq_length
    assert len(train_data_processor.label_ids[0]) == max_seq_length

    print("Sequense train len = ", len(train_data_processor.input_ids[0]))
    print("Num of sequences = ", len(train_data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    val_data_processor = GetIndices(ftype='val', data_dir='./new_data_split/')
    val_data_processor.upload_hdf()
    assert len(val_data_processor.input_ids[0]) == max_seq_length
    assert len(val_data_processor.input_mask[0]) == max_seq_length
    assert len(val_data_processor.label_ids[0]) == max_seq_length

    print("Sequense val len = ", len(val_data_processor.input_ids[0]))
    print("Num of sequences = ", len(val_data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    #model = TsyaModel(TrainProcessor = TrainProcessor, ValProcessor = ValProcessor)
    model = TsyaModel(weight_path=None, train_from_chk=False)
    model.train(chkp_path, train_data_processor, val_data_processor)


if __name__ == "__main__":
    main()
    # print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))
    # print(torch.cuda.is_available())