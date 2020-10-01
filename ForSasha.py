from Class import TestPreprocess, ProcessOutput
from Model import TsyaModel
import json
from transformers import BertTokenizer, AutoTokenizer


class OrphoNet:
    
    def __init__(self):
        with open("config_stand.json") as json_data_file:
            data = json.load(json_data_file)
        if not data['from_rubert']:
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(**data['config_of_tokenizer'])
        self.output = ProcessOutput(tokenizer=tokenizer)
        self.data_processor = TestPreprocess(tokenizer=tokenizer)
        self.model = TsyaModel(weight_path="Chkpts/"+data['weight_path'], train_from_chk=data['train_from_chk'],
                               label_list=data['label_list'], seed_val=data["seed_val"],
                               from_rubert=data['from_rubert'], adam_options=data["adam_options"], tokenizer=tokenizer,
                               config_of_model=data["config_of_model"])

    def execute(self, sentences):
        data_with_tsya_or_nn = self.data_processor.check_contain_tsya_or_nn([sentences])
        if len(data_with_tsya_or_nn) == 0:
            message = ["Correct"]
            return message
        else:
            input_ids, mask_ids, prediction_dataloader, nopad = self.data_processor.process(text=data_with_tsya_or_nn)

            predicts, probabilities, probabilities_o = self.model.predict_batch(prediction_dataloader, nopad)
            # message, _, correct_text, _, _, _, correction_dict = self.output.process_sentence(
            #             predicts, input_ids, nopad, data_with_tsya_or_nn, probabilities, probabilities_o,
            #             default_value='Correct', threshold=0.5)
            correct_text, correction_dict, words_errors, words_probs = self.output.process_sentence_optimal(
                        predicts, input_ids, nopad, data_with_tsya_or_nn, probabilities, probabilities_o,
                        default_value='Correct', threshold=0.5)
            if len(correction_dict.keys()) > 0:
                message = "Incorrect"
                return [message, correct_text, correction_dict, words_errors, words_probs]
            else:
                message = "Correct"
                return [message]

    def give_json(self, correction_dict, file_name):

        with open(file_name, 'w', encoding='utf-8') as outfile:
            json.dump(correction_dict, outfile, ensure_ascii=False, indent=4)

    def execute_old(self, sentences, default_value):
        data_with_tsya_or_nn = self.data_processor.check_contain_tsya_or_nn([sentences])
        if len(data_with_tsya_or_nn) == 0:
            message = [[["Correct"]]]
            return message
        else:
            input_ids, mask_ids, prediction_dataloader, nopad = self.data_processor.process(text=data_with_tsya_or_nn)

            predicts, probabilities, probabilities_o = self.model.predict_batch(prediction_dataloader, nopad)
            message, incorrect_words, correct_text, error, probs, probs_O, _ = self.output.process_sentence(
                        predicts, input_ids, nopad, data_with_tsya_or_nn, probabilities, probabilities_o,
                        default_value, threshold=0.5)
            return message, incorrect_words, correct_text, error, probs, probs_O

# text_data = 'мощёная каменными плитами дорога'
# text_data = "Если находится внутри задачи, то написать что-то в форму обратной связи становится невозможно."
# text_data = "Можно собрать эти сведения вручную, автоматизировано или используя оба способа."
# text_data = "Длинна рамки, используемой для поиска компактных вхождений"
# text_data = "Пете Иванову явится с вещами"
# text_data = "Остались считанные минуты"
# text_data = "Решен только вопрос о том, что он получил документ, что его заявление по предоставлению убежища получен\
# ФМС и будет рассмотрен в установленом законом порядке."
# model = OrphoNet()
# text_data = "Бонне, Шарлотта\n"
# output = model.execute(text_data)
# # model.give_json(output[2], 'model_output.json')
# for out in output:
#     print(out)
