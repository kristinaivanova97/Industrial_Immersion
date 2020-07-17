import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import re
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
weight_path = "Chkpt.pth"
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestPreprocess:
    
    def __init__(self, tokenizer = tokenizer):
        
        label_list = ["[Padding]", "[SEP]", "[CLS]", "O","ться", "тся"]
        
        self.label_map = {}
        for (i, label) in enumerate(label_list):
            self.label_map[label] = i
            
        self.input_ids = []
        self.attention_masks = []
        self.nopad = []
        self.tokenizer = tokenizer
        
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
            self.nopad.append(len(ntokens))
            label_ids.append(self.label_map["[SEP]"])
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
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
            
        self.input_ids = torch.tensor(input_ids_full)
        self.attention_masks = torch.tensor(attention_masks)
        prediction_data = TensorDataset(self.input_ids, self.attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        return self.input_ids, self.attention_masks, prediction_dataloader, self.nopad, label_ids_full
    
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
                elif (m2 is not None):
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
    def __init__(self, weight_path = weight_path, tokenizer = tokenizer):
        
        label_list = ["[Padding]", "[SEP]", "[CLS]", "O","ться", "тся"]
        
        self.m =  BertForTokenClassification.from_pretrained(
                        'bert-base-multilingual-cased',
                        num_labels = len(label_list),
                        output_attentions = False,
                        output_hidden_states = False,
                    )
        self.m.load_state_dict(torch.load(weight_path))
        self.m.to(device)
        self.tokenizer = tokenizer
        
    def predict_batch(self, prediction_dataloader, nopad):
        self.m.eval()
        predicts_full = []
        sentences_full = []
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
