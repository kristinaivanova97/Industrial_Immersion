import json
from datetime import datetime
from pathlib import Path

from transformers import BertTokenizer, AutoTokenizer

from Model import GetIndices, TsyaModel
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def main():

    with open("config_train.json") as json_data_file:
        configs = json.load(json_data_file)

    train_data_processor = GetIndices(ftype=configs["train_file"], data_dir=configs["data_path"])
    train_data_processor.upload_hdf()
    assert len(train_data_processor.input_ids[0]) == configs["max_seq_length"]
    assert len(train_data_processor.input_mask[0]) == configs["max_seq_length"]
    assert len(train_data_processor.label_ids[0]) == configs["max_seq_length"]

    print("Sequense train len = ", len(train_data_processor.input_ids[0]))
    print("Num of sequences = ", len(train_data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    val_data_processor = GetIndices(ftype=configs["val_file"], data_dir=configs["data_path"])
    val_data_processor.upload_hdf()
    assert len(val_data_processor.input_ids[0]) == configs["max_seq_length"]
    assert len(val_data_processor.input_mask[0]) == configs["max_seq_length"]
    assert len(val_data_processor.label_ids[0]) == configs["max_seq_length"]

    print("Sequense val len = ", len(val_data_processor.input_ids[0]))
    print("Num of sequences = ", len(val_data_processor.input_ids))
    print("files with input ids, masks, segment ids and label ids are loaded succesfully")

    if not configs['from_rubert']:
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    else:
        tokenizer = AutoTokenizer.from_pretrained(**configs['config_of_tokenizer'])

    label_list = configs["label_list"]
    if configs["part_of_word"]:
        label_list.append("[##]")

    model = TsyaModel(label_list=label_list, weight_path=None, train_from_chk=configs["train_from_chk"],
                      seed_val=configs['seed_val'], tokenizer=tokenizer, from_rubert=configs['from_rubert'],
                      config_of_model=configs['config_of_model'], adam_options=configs['adam_options'])

    Path(configs["weight_path"]).mkdir(parents=True, exist_ok=True)
    number_of_chkp = 0
    path_to_file_chkp = configs["weight_path"] + configs["chckp_file"] \
                        + str(datetime.date(datetime.now()).year)[-2:] + str(datetime.date(datetime.now()).month) \
                        + str(datetime.date(datetime.now()).day) \
                        + "_" + str(number_of_chkp)
    while Path(path_to_file_chkp + ".pth").is_file():
        number_of_chkp+=1
        path_to_file_chkp = configs["weight_path"] + configs["chckp_file"] \
                            + str(datetime.date(datetime.now()).year)[-2:] + str(datetime.date(datetime.now()).month) \
                            + str(datetime.date(datetime.now()).day) \
                            + "_" + str(number_of_chkp)

    model.train(train_data_processor=train_data_processor, val_data_processor=val_data_processor,
                chkp_path=path_to_file_chkp + ".pth", epochs=configs["epochs"],
                batch_size=configs["batch_size"], do_validation=True)


if __name__ == "__main__":
    main()
