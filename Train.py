import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from Model import GetIndices, TsyaModel

batch_size = 1
epochs = 3 # The BERT authors recommend between 2 and 4.
max_seq_length = 512 # for bert this limit exists
data_dir = "./new_data/"
chkp_path = "Chkpt_test.pth"
# input indices are created in another file (DataPreprocess.py) - essential for Bert model

def main():

    data_processor = GetIndices(ftype = 'data')
    data_processor.upload()
    assert len(data_processor.input_ids[0]) == max_seq_length
    assert len(data_processor.input_mask[0]) == max_seq_length
    assert len(data_processor.label_ids[0]) == max_seq_length

    print("Sequense len = ", len(data_processor.input_ids[0]))
    print("Num of sequences = ", len(data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    #model = TsyaModel(TrainProcessor = TrainProcessor, ValProcessor = ValProcessor)
    model = TsyaModel()
    model.train(chkp_path, data_processor)


if __name__ == "__main__":
    main()

