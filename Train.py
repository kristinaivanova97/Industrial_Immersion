from Model import GetIndices, TsyaModel

max_seq_length = 512 # for bert this limit exists
data_dir = "./data_split/"
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
trainprocessor = GetIndices(ftype='train', data_dir=data_dir)
valprocessor = GetIndices(ftype='val', data_dir=data_dir)
trainprocessor.upload_hdf()
valprocessor.upload_hdf()

assert len(trainprocessor.input_ids[0]) == max_seq_length
assert len(trainprocessor.input_mask[0]) == max_seq_length
assert len(trainprocessor.label_ids[0]) == max_seq_length

print("Sequense len = ", len(trainprocessor.input_ids[0]))
print("Num of sequences = ", len(trainprocessor.input_ids))
print("Num of val sequences = ", len(valprocessor.input_ids))
print("files with input ids, masks, segment ids and label ids are loaded succesfully")

model = TsyaModel()
print("Initialisation is fininshed")
#model.train(chkp_path, DataProcessor)
model.train(chkp_path, trainprocessor, valprocessor)
