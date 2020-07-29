from Train_Classes import GetIndices, TsyaModel

max_seq_length = 512 # for bert this limit exists
data_dir = "./new_data/"
chkp_path = "Chkpt_part_of_word.pth"

'''
DataProcessor = GetIndices(ftype = 'data', data_dir = data_dir)
DataProcessor.upload()
assert len(DataProcessor.input_ids[0]) == max_seq_length
assert len(DataProcessor.input_mask[0]) == max_seq_length
assert len(DataProcessor.label_ids[0]) == max_seq_length

print("Sequense len = ", len(DataProcessor.input_ids[0]))
print("Num of sequences = ", len(DataProcessor.input_ids))
print("files with input ids, masks, segment ids and label ids are loaded succesfully")
'''

TrainProcessor = GetIndices(ftype = 'train', data_dir = data_dir)
ValProcessor = GetIndices(ftype = 'val', data_dir = data_dir)
TrainProcessor.upload()
ValProcessor.upload()

assert len(TrainProcessor.input_ids[0]) == max_seq_length
assert len(TrainProcessor.input_mask[0]) == max_seq_length
assert len(TrainProcessor.label_ids[0]) == max_seq_length

print("Sequense len = ", len(TrainProcessor.input_ids[0]))
print("Num of sequences = ", len(TrainProcessor.input_ids))
print("Num of val sequences = ", len(ValProcessor.input_ids))
print("files with input ids, masks, segment ids and label ids are loaded succesfully")

#model = TsyaModelTrain(TrainProcessor = TrainProcessor, ValProcessor = ValProcessor)
model = TsyaModel()
print("Initialisation is fininshed")
model.train(chkp_path, DataProcessor)
