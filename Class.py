import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import re
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertConfig
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
weight_path = "chkpt.pth"
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
        for sentence in tqdm(text):
            tokens = []
            for i, word in enumerate(sentence.split()):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
            ntokens = []
            ntokens.append("[CLS]")
            for i, token in enumerate(tokens):
                ntokens.append(token)

            ntokens.append("[SEP]")
            self.nopad.append(len(ntokens))
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                ntokens.append("[Padding]")
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            input_ids_full.append(input_ids)
            attention_masks.append(input_mask)
        self.input_ids = torch.tensor(input_ids_full)
        self.attention_masks = torch.tensor(attention_masks)
        prediction_data = TensorDataset(self.input_ids, self.attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        return self.input_ids, self.attention_masks, prediction_dataloader, self.nopad
        

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
                #print(tokenizer.convert_ids_to_tokens(b_input_ids[i,:nopad[i*step + i]]))
                predicts.append(prediction[i, :nopad[step]])
                #print(prediction[i, :nopad[i*step + i]])
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
