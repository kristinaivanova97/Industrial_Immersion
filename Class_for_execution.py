from Class import TestPreprocess, ProcessOutput
from Model_lexi import TsyaModel
import json
from transformers import BertTokenizer


class OrphoNet:
    
    def __init__(self):
        with open("config_stand.json") as json_data_file:
            self.configs = json.load(json_data_file)
        with open("test.json") as json_data_file:
            tags = json.load(json_data_file)

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        self.output = ProcessOutput(tokenizer=tokenizer)
        self.data_processor = TestPreprocess(tokenizer=tokenizer)
        self.model = TsyaModel(weight_path=self.configs['weight_path'] + self.configs['chckp_file'],
                               train_from_chk=self.configs['train_from_chk'],
                               label_list=self.configs['label_list'], seed_val=self.configs["seed_val"],
                               from_rubert=self.configs['from_rubert'], adam_options=self.configs["adam_options"],
                               tokenizer=tokenizer, config_of_model=self.configs["config_of_model"], multilingual=False)

    def execute(self, sentences, default_value='Correct'):

        input_ids, mask_ids, prediction_dataloader, nopad = \
            self.data_processor.process(text=[sentences], max_seq_length=self.configs["max_seq_len"])

        predicts, probabilities, probabilities_o = self.model.predict_batch(prediction_dataloader, nopad)
        correct_text, correction_dict, words_errors, incorrect_words, corrected_words = \
            self.output.process_sentence_optimal(
                    predicts, input_ids, nopad, [sentences], probabilities, probabilities_o,
                    default_value, threshold=0.5, check_in_dict=self.configs["check_in_dict"])
        if len(correction_dict.keys()) > 0:
            message = "Incorrect"
            return [message, correct_text, correction_dict, words_errors, incorrect_words, corrected_words]
        else:
            message = "Correct"
            return [message]

    def give_json(self, correction_dict, file_name):

        with open(file_name, 'w', encoding='utf-8') as outfile:
            json.dump(correction_dict, outfile, ensure_ascii=False, indent=4)

# text_data = 'мощёная каменными плитами дорога'
# text_data = "Если находится внутри задачи, то написать что-то в форму обратной связи становится невозможно."
# text_data = "Можно собрать эти сведения вручную, автоматизировано или используя оба способа."
# text_data = "Длинна рамки, используемой для поиска компактных вхождений"
# text_data = "Пете Иванову явится с вещами"
# text_data = "Остались считанные минуты"
# text_data = "Решен только вопрос о том, что он получил документ, что его заявление по предоставлению убежища получен\
# ФМС и будет рассмотрен в установленом законом порядке."


model = OrphoNet()
text_data = "Others argue that there are no more important things than friends and relatives in existant of each man."
output = model.execute(text_data, 'Incorrect')
# model.give_json(output[2], 'model_output.json')
for out in output:
    print(out)

