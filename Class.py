import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import re
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

weight_path = "Chkpt.pth"
batch_size = 16
max_seq_length = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestPreprocess:
    
    def __init__(self):

        self.label_list = ["[Padding]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya",
                           "REPLACE_tsya",
                           "[##]"]
        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        
    def process(self, text, max_seq_length = max_seq_length, batch_size = batch_size):

        input_ids_full = []
        attention_masks = []
        # label_ids_full = []
        nopad = []

        # y_label = self.gettags(text)
        for i, sentence in enumerate(text):
            tokens = []
            labels = []
            for j, word in  enumerate(re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                # label_1 = y_label[i][j]
                # for m in range(len(token)):
                #     labels.append(label_1)

            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
            ntokens = []
            label_ids = []
            ntokens.append("[CLS]")
            label_ids.append(self.label_map["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                label_ids.append(self.label_map[labels[i]])
                    
            ntokens.append("[SEP]")
            nopad.append(len(ntokens))
            label_ids.append(self.label_map["[SEP]"])
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                label_ids.append(0)
                ntokens.append("[Padding]")
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            input_ids_full.append(input_ids)
            attention_masks.append(input_mask)
            # label_ids_full.append(label_ids)
            
        input_ids = torch.tensor(input_ids_full)
        attention_masks = torch.tensor(attention_masks)
        prediction_data = TensorDataset(input_ids, attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        return input_ids, attention_masks, prediction_dataloader, nopad
    
    # def gettags(self, text):
    #
    #     tsya_search = re.compile(r'тся\b')
    #     tsiya_search = re.compile(r'ться\b')
    #     dicty = {}
    #     i = 0
    #     for raw in tqdm(text):
    #
    #         m = tsya_search.findall(raw)
    #         m2 = tsiya_search.findall(raw)
    #
    #         for j, word in  enumerate(re.findall(r'\w+|[^\w\s]', raw, re.UNICODE)):
    #
    #             m = tsya_search.search(word)
    #             m2 = tsiya_search.search(word)
    #             dicty.setdefault(i, {})
    #             if m is not None:
    #                 dicty[i][j] = m.group() # "тся" label
    #             elif m2 is not None:
    #                 dicty[i][j] = m2.group() # "ться" label
    #             else:
    #                 dicty[i][j] = "O"
    #         i+=1
    #
    #     y_label = []
    #     for i in dicty.keys():
    #         raw = []
    #         for j in range(len(dicty[i])):
    #             raw.append(dicty[i][j])
    #         y_label.append(raw)
    #     return y_label


# class TsyaModel:
#
#     def __init__(self, weight_path = weight_path):
#
#         label_list = ["[Padding]", "[SEP]", "[CLS]", "O","ться", "тся"]
#         self.m =  BertForTokenClassification.from_pretrained(
#                         'bert-base-multilingual-cased',
#                         num_labels = len(label_list),
#                         output_attentions = False,
#                         output_hidden_states = False,
#                     )
#         self.m.load_state_dict(torch.load(weight_path))
#         self.m.to(device)
#
#     def predict_batch(self, prediction_dataloader, nopad):
#
#         self.m.eval()
#         predicts_full = []
#         step = 0
#         for batch in prediction_dataloader:
#             batch = tuple(t.to(device) for t in batch)
#             b_input_ids, b_input_mask = batch
#             with torch.no_grad():
#
#                 output = self.m(b_input_ids, token_type_ids=None,
#                               attention_mask=b_input_mask)
#             logits = output[0].detach().cpu().numpy()
#             prediction = np.argmax(logits, axis=2)
#             predicts = []
#             for i in range(len(b_input_ids)):
#
#                 predicts.append(prediction[i, :nopad[step]])
#                 step+=1
#             predicts_full.append(predicts)
#
#         return predicts_full
#
#     def predict_sentence(self, input_ids, attention_masks, nopad):
#
#         self.m.eval()
#         predicts = []
#         input_ids = input_ids.to(device)
#         input_mask = attention_masks.to(device)
#
#         with torch.no_grad():
#                 output = self.m(input_ids, token_type_ids=None,
#                               attention_mask=input_mask)
#         logits = output[0].detach().cpu().numpy()
#         prediction = np.argmax(logits, axis=2)
#         predicts.append(prediction[0, :nopad[0]])
#         return predicts


class ProcessOutput:
    
    def __init__(self):
        
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        self.tokens = []
        self.text = []
        self.fine_text = ''
        self.tags = []
        self.preds = []
        self.correct_text = ''

    def print_results_in_file(self, file_name):
        print("Tokens = ", self.tokens, file=file_name)
        print("Prediction = ", self.preds, file=file_name)
        print("Initial Tags = ", self.tags, file=file_name)
        print("Fine text = {} \n".format(self.fine_text), file=file_name)
        print("Correct text = {} \n".format(self.correct_text), file=file_name)
        print(file=file_name)

    def print_results(self):
        print("Tokens = ", self.tokens)
        print("Prediction = ", self.preds)
        print("Initial Tags = ", self.tags)
        print("Fine text = {} \n".format(self.fine_text))
        print("Correct text = {} \n".format(self.correct_text))

    def process(self, predictions, input_ids, nopad, text_data):

        # with open('results.txt', 'w') as file_name:
        step = 0
        for i,predict in enumerate(predictions):
            for j, pred in enumerate(predict):
                self.tokens = self._tokenizer.convert_ids_to_tokens(input_ids[step, :nopad[step]])
                self.text = self._tokenizer.decode(input_ids[step, :nopad[step]])
                # self.fine_text = self.text.replace('[CLS] ', '').replace(' [SEP]', '')
                self.fine_text = text_data[step]
                # self.tags =  np.array(data_tags[step][:nopad[step]])
                self.preds = np.array(pred)
                self.correct_text = self.fine_text
                all_incorrect = []
                message = ["Correct"]

                if self._check_coincide() > 0:
                    message = ["Incorrect"]
                    incorrect_words_tisya = []
                    incorrect_words_tsya = []

                    array_of_tokens_with_tisya = np.where(self.tags==4)[0].tolist()
                    array_of_tokens_with_tsya = np.where(self.tags==5)[0].tolist()


                    if len(array_of_tokens_with_tisya) > 0:
                        array_of_word_indexes = [array_of_tokens_with_tisya[0]]
                        word = self.tokens[array_of_tokens_with_tisya[0]]
                        for index in array_of_tokens_with_tisya[1:]:
                            if '##' in self.tokens[index]:
                                array_of_word_indexes.append(index)
                                word += self.tokens[index][2:]
                            else:
                                if 5 in self.preds[array_of_word_indexes[0]:array_of_word_indexes[-1]+1]:
                                    incorrect_words_tisya.append(word)
                                    all_incorrect.append(word)
                                word = self.tokens[index]
                        if 5 in self.preds[array_of_word_indexes[0]:array_of_word_indexes[-1] + 1]:
                            incorrect_words_tisya.append(word)
                            all_incorrect.append(word)

                    if len(array_of_tokens_with_tsya)>0:
                        array_of_word_indexes = [array_of_tokens_with_tsya[0]]
                        word = self.tokens[array_of_tokens_with_tsya[0]]
                        for index in array_of_tokens_with_tsya[1:]:
                            if '##' in self.tokens[index]:
                                array_of_word_indexes.append(index)
                                word += self.tokens[index][2:]
                            else:
                                if 4 in self.preds[array_of_word_indexes[0]:array_of_word_indexes[-1]+1]:
                                    incorrect_words_tsya.append(word)
                                    all_incorrect.append(word)
                                word = self.tokens[index]

                        if 4 in self.preds[array_of_word_indexes[0]:array_of_word_indexes[-1]+1]:
                            incorrect_words_tsya.append(word)
                            all_incorrect.append(word)

                    for word in incorrect_words_tisya:
                        word_correct = word.replace('ться', 'тся')
                        self.correct_text = self.correct_text.replace(word, word_correct)

                    for word in incorrect_words_tsya:
                        word_correct = word.replace('тся', 'ться')
                        self.correct_text = self.correct_text.replace(word, word_correct)

                self.print_results()

                step+=1
        return all_incorrect, message, self.correct_text

    def _check_coincide(self):
        
        coincide = np.sum(self.tags[(self.tags==4) | (self.tags==5)] == self.preds[(self.tags==4) | (self.tags==5)])
        #print("Coincide in {} positions with tsya/tsiya ".format(coincide))
        if coincide == len(self.tags[(self.tags==4) | (self.tags==5)]):
            if (len(self.tags[(self.tags==4) | (self.tags==5)]) == 0):
                print("Sentence does not contain words with tsya/tsiya")
            else:
                print("Predicted and initial sentences coincide")
            return 0
        else:
            print("Sentence contain a mistake!")
            return 1
