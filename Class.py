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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestPreprocess:
    
    def __init__(self):
        
        label_list = ["[Padding]", "[SEP]", "[CLS]", "O","ться", "тся"]
        
        self.label_map = {}
        for (i, label) in enumerate(label_list):
            self.label_map[label] = i
            
        self._input_ids = []
        self._attention_masks = []
        self._nopad = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
        
    def process(self, text, max_seq_length = 512, batch_size = batch_size):

        input_ids_full = []
        attention_masks = []
        label_ids_full = []
        
        y_label = self.__GetTags(text)
        for i, sentence in enumerate(text):
            tokens = []
            labels = []
            for j, word in  enumerate(re.findall(r'\w+|[^\w\s]', sentence, re.UNICODE)):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = y_label[i][j]
                for m in range(len(token)):
                    labels.append(label_1)
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
            self._nopad.append(len(ntokens))
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
            label_ids_full.append(label_ids)
            
        self._input_ids = torch.tensor(input_ids_full)
        self._attention_masks = torch.tensor(attention_masks)
        prediction_data = TensorDataset(self._input_ids, self._attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        return self._input_ids, self._attention_masks, prediction_dataloader, self._nopad, label_ids_full
    
    def __GetTags(self, text):
        
        tsya_search = re.compile(r'тся\b')
        tsiya_search = re.compile(r'ться\b')
        dicty = {}
        i = 0
        for raw in tqdm(text):
        
            m = tsya_search.findall(raw)
            m2 = tsiya_search.findall(raw)

            for j, word in  enumerate(re.findall(r'\w+|[^\w\s]', raw, re.UNICODE)):

                m = tsya_search.search(word)
                m2 = tsiya_search.search(word)
                dicty.setdefault(i, {})
                if m is not None:
                    dicty[i][j] = m.group() # "тся" label
                elif m2 is not None:
                    dicty[i][j] = m2.group() # "ться" label
                else:
                    dicty[i][j] = "O"
            i+=1

        y_label = []
        for i in dicty.keys():
            raw = []
            for j in range(len(dicty[i])):
                raw.append(dicty[i][j])
            y_label.append(raw)
        return y_label

class TsyaModel:
    def __init__(self, weight_path = weight_path):
        
        label_list = ["[Padding]", "[SEP]", "[CLS]", "O","ться", "тся"]
        self.m =  BertForTokenClassification.from_pretrained(
                        'bert-base-multilingual-cased',
                        num_labels = len(label_list),
                        output_attentions = False,
                        output_hidden_states = False,
                    )
        self.m.load_state_dict(torch.load(weight_path))
        self.m.to(device)
        
    def predict_batch(self, prediction_dataloader, nopad):
        
        self.m.eval()
        predicts_full = []
        step = 0
        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():

                output = self.m(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask)
            logits = output[0].detach().cpu().numpy()
            prediction = np.argmax(logits, axis=2)
            predicts = []
            for i in range(len(b_input_ids)):

                predicts.append(prediction[i, :nopad[step]])
                step+=1
            predicts_full.append(predicts)
            
        return predicts_full
    
    def predict_sentence(self, input_ids, attention_masks, nopad):
        
        self.m.eval()
        predicts = []
        input_ids = input_ids.to(device)
        input_mask = attention_masks.to(device)
        
        with torch.no_grad():
                output = self.m(input_ids, token_type_ids=None,
                              attention_mask=input_mask)
        logits = output[0].detach().cpu().numpy()
        prediction = np.argmax(logits, axis=2)
        predicts.append(prediction[0, :nopad[0]])
        return predicts

class ProcessOutput:
    
    def __init__(self):
        
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
    
    def process(self, predictions, input_ids, nopad, data_tags):
    
        
        if len(predictions) < 2:
        
            toks = self._tokenizer.convert_ids_to_tokens(input_ids[0, :nopad[0]])
            text = self._tokenizer.decode(input_ids[0, :nopad[0]])
            fine_text = text.replace('[CLS] ', '').replace(' [SEP]', '')
            tags = np.array(data_tags[0][:nopad[0]])
            preds =  np.array(list(predictions[0]))

            self.__check_coincide(tags, preds)

            print("Tokens = ", toks)
            print("Prediction = ", preds)
            print("Initial Tags = ", tags)
            print("Fine text = {} \n".format(fine_text))

        else:
            step = 0
            for i,predict in enumerate(predictions):
                for j, pred in enumerate(predict):
                    toks = self._tokenizer.convert_ids_to_tokens(input_ids[step, :nopad[step]])
                    text = self._tokenizer.decode(input_ids[step, :nopad[step]])
                    fine_text = text.replace('[CLS] ', '').replace(' [SEP]', '')
                    nomask_pred = pred[1:-1]
                    tags =  np.array(data_tags[step][:nopad[step]])
                    preds = np.array(pred)

                    self.__check_coincide(tags, preds)

                    print("Tokens = ", toks)
                    print("Prediction = ", preds)
                    print("Initial Tags = ", tags)

                    print("Fine text = {} \n".format(fine_text))
                    step+=1
                    
    def __check_coincide(self, tags, preds):
        
        coincide = np.sum(tags[(tags==4) | (tags==5)] == preds[(tags==4) | (tags==5)])
        print("Coincide in {} positions with tsya/tsiya ".format(coincide))
        if coincide == len(tags[(tags==4) | (tags==5)]):
            if (len(tags[(tags==4) | (tags==5)]) == 0):
                print("Sentence does not contain words with tsya/tsiya")
            else:
                print("Predicted and initial sentences coincide")
        else:
            print("Sentence contain a mistake!")
