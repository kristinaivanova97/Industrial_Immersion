import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import random

batch_size = 7
epochs = 3 # The BERT authors recommend between 2 and 4.
max_seq_length = 512 # for bert this limit exists

data_dir = "./data_split/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cuda'
print(torch.__version__)
print("Device: {}".format(device))
weight_path = "Chkpt_test.pth"


class GetIndices:

    def __init__(self, ftype, data_dir):
        self.file_names = [data_dir + 'input_ids_' + ftype + '.txt', data_dir + 'input_mask_' + ftype + '.txt', data_dir + 'label_ids_' + ftype + '.txt']
        self.file_hdf = data_dir + ftype + '.hdf5'
        self.input_ids = []
        self.input_mask = []
        self.label_ids = []


    def upload(self):

        features = [self.input_ids, self.input_mask, self.label_ids]
        for i in tqdm(range(len(self.file_names))):
            my_file = open(self.file_names[i], 'r')
            lines = my_file.readlines()
            list_of_lists = []

            for line in lines:
                stripped_line = line.strip()
                line_list = stripped_line.split()
                list_of_lists.append(line_list)


            my_file.close()
            for j in range(len(list_of_lists)):
                features[i].append(list(map(int, list_of_lists[j])))

    def upload_hdf(self):

        with h5py.File(self.file_hdf, 'r') as f:
            self.input_ids = f['input_ids'][:,:]
            self.input_mask = f['input_mask'][:,:]
            self.label_ids = f['label_ids'][:,:]


    # def get_labels(self, filename):
    #
    #     # *** not nessesary as wont't be used later ***
    #     my_file = open(filename, 'r') # 'Labels.txt'
    #     y_label = []
    #     for line in my_file:
    #         stripped_line = line.strip()
    #         line_list = stripped_line.split()
    #         y_label.append(line_list)
    #     my_file.close()
    #     print("Size of y_label = ", len(y_label))
    #     print("*** pleminary labels are created ***")
    #
    #     return y_label

class TsyaModel:


    def __init__(self, weight_path=None, train_from_chk=False, device=device):

        self.weight_path = weight_path
        self.label_list = ["[Padding]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya", "REPLACE_tsya",
                      "[##]"]
        self.label_map = {}
        for (i, label) in enumerate(self.label_list):
            self.label_map[label] = i

        self.model = BertForTokenClassification.from_pretrained(
            'bert-base-multilingual-cased',
            num_labels=len(self.label_list),
            output_attentions=False,
            output_hidden_states=False,
        )

        if train_from_chk:
            self.model.load_state_dict(torch.load(self.weight_path))

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

        self.optimizer = AdamW(self.model.parameters(),
                          lr = 2e-5, # args.learning_rate - default is 5e-5
                          eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                         )
        self.model.to(device)
        self.seed_val = 42

    def format_time(self, elapsed):

        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):

        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat[labels_flat != 0] == labels_flat[labels_flat != 0]) / len(labels_flat[labels_flat != 0])


    def _dataset(self, train_processor, val_processor):

        # Combine the training inputs into a TensorDataset.

        dataset = TensorDataset(torch.tensor(train_processor.input_ids[:15000]),
                                torch.tensor(train_processor.input_mask[:15000]),
                                torch.tensor(train_processor.label_ids[:15000]))

        val_dataset = TensorDataset(torch.tensor(val_processor.input_ids[:20000]),
                                    torch.tensor(val_processor.input_mask[:20000]),
                                    torch.tensor(val_processor.label_ids[:20000]))

        train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
        validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

        return train_dataloader, validation_dataloader

    def _new_dataset(self, data_processor):

        dataset = TensorDataset(torch.tensor(data_processor.input_ids[:150]),
                                torch.tensor(data_processor.input_mask[:150]),
                                torch.tensor(data_processor.label_ids[:150]))

        val_dataset = TensorDataset(torch.tensor(data_processor.input_ids[150:200]),
                                    torch.tensor(data_processor.input_mask[150:200]),
                                    torch.tensor(data_processor.label_ids[150:200]))

        train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
        print("Train loader has been loaded")
        validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

        return train_dataloader, validation_dataloader
    
    def _from_hdf5_dataset(self, train_processor, val_processor):
        
        dataset = TensorDataset(torch.from_numpy(train_processor.input_ids),
                                torch.from_numpy(train_processor.input_mask),
                                torch.from_numpy(train_processor.label_ids))

        val_dataset = TensorDataset(torch.from_numpy(val_processor.input_ids),
                                    torch.from_numpy(val_processor.input_mask),
                                    torch.from_numpy(val_processor.label_ids))

        train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
        validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

        return train_dataloader, validation_dataloader


    def format_time(self, elapsed):

        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):

        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat[labels_flat != 0] == labels_flat[labels_flat != 0]) / len(labels_flat[labels_flat != 0])


    #def train(self, chkp_path, data_processor):
    def train(self, chkp_path, train_processor, val_processor):

        #self.train_dataloader, self.validation_dataloader = self._new_dataset(data_processor=data_processor)
        self.train_dataloader, self.validation_dataloader = self._from_hdf5_dataset(train_processor=train_processor, val_processor=val_processor)
        print("Dataloader is created")
        
        total_steps = len(self.train_dataloader) * epochs

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=total_steps)

        self.params = list(self.model.named_parameters())

        print('==== Embedding Layer ====\n')

        for p in self.params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in self.params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

        training_stats = []
        total_t0 = time.time()

        for epoch_i in range(epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0

            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader),
                                                                                elapsed))

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                print("Labels ", b_labels.shape)
		
                self.model.zero_grad()

                loss, logits = self.model(b_input_ids,
                                          token_type_ids=None,  # b_segment,
                                          attention_mask=b_input_mask,
                                          labels=b_labels.to(dtype = torch.long))
                
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()
            avg_train_loss = total_train_loss / len(self.train_dataloader)

            training_time = self.format_time(time.time() - t0)

            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t0 = time.time()

            self.model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in self.validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    (loss, logits) = self.model(b_input_ids,
                                                token_type_ids=None,  # b_segment,
                                                attention_mask=b_input_mask,
                                                labels=b_labels.to(dtype = torch.long))

                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                print(label_ids.shape, logits.shape)
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)
            print(self.tokenizer.convert_ids_to_tokens(b_input_ids[0, :]))
            last = np.argmax(logits, axis=2)
            part = last[0, :]
            part_true = label_ids[0, :]
            print("Last true = ", part_true)
            print("Last prediction", part)

            avg_val_accuracy = total_eval_accuracy / len(self.validation_dataloader)
            print("  Accuracy: {0:.3f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.validation_dataloader)

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.3f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

            print("  Average training loss: {0:.3f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))
            torch.save(self.model.state_dict(), chkp_path)
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time() - total_t0)))
        torch.save(self.model.state_dict(), chkp_path)



    def predict(self, prediction_dataloader, nopad):
        self.model.eval()
        predicts_full = []
        step = 0
        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():

                output = self.model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask)
            logits = output[0].detach().cpu().numpy()
            prediction = np.argmax(logits, axis=2)
            predicts = []
            for i in range(len(b_input_ids)):
                predicts.append(prediction[i, :nopad[step]])
                step += 1
            predicts_full.append(predicts)

        return predicts_full




# input indices are created in another file (DataPreprocess.py) - essential for Bert model
'''
class GetIndices:
    
    def __init__(self, ftype, data_dir = data_dir):
        
        self.file_names = [data_dir + 'input_ids_' + ftype + '.txt', data_dir + 'input_mask_' + ftype + '.txt', data_dir + 'label_ids_' + ftype + '.txt']
        self.input_ids = []
        self.input_mask = []
        self.label_ids = []
        
    def upload(self):
        
        features = [self.input_ids, self.input_mask, self.label_ids]
        for i in tqdm(range(len(self.file_names))):
            my_file = open(self.file_names[i], 'r')
            list_of_lists = []
            for line in my_file:
                stripped_line = line.strip()
                line_list = stripped_line.split()
                list_of_lists.append(line_list)

            my_file.close()
            for j in range(len(list_of_lists)):
                features[i].append(list(map(int, list_of_lists[j])))   

    
    def getlabels(self, filename):
        
        # *** not nessesary as wont't be used later ***
        my_file = open(filename, 'r') # 'Labels.txt'
        y_label = []
        for line in my_file:
            stripped_line = line.strip()
            line_list = stripped_line.split()
            y_label.append(line_list)
        my_file.close()
        print("Size of y_label = ", len(y_label))
        print("*** pleminary labels are created ***")
        
        return y_label
'''
'''
class TsyaModelTrain:
    
    #def __init__(self, TrainProcessor, ValProcessor, epochs = epochs, device = device):
    def __init__(self, DataProcessor, epochs = epochs, device = device):
        #label_list = ["[Padding]", "[SEP]", "[CLS]", "O","ться", "тся"] # all possible labels
        label_list = ["[Padding]", "[SEP]", "[CLS]", "O", "REPLACE_nn", "REPLACE_n", "REPLACE_tysya", "REPLACE_tsya",
                      "[##]"]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
        self.model = BertForTokenClassification.from_pretrained(
            'bert-base-multilingual-cased', # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = len(label_list), # The number of output labels   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        self.model.to(device)

        self.optimizer = AdamW(self.model.parameters(),
                          lr = 2e-5, # args.learning_rate - default is 5e-5
                          eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                         )
        
        #self.train_dataloader, self.validation_dataloader = self.__dataset(TrainProcessor = TrainProcessor, ValProcessor = ValProcessor)
        
        self.train_dataloader, self.validation_dataloader = self.__new_dataset(DataProcessor = DataProcessor)
        print("Dataloader is created")
           
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(self.train_dataloader) * epochs

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
        
        self.params = list(self.model.named_parameters())

        print('==== Embedding Layer ====\n')

        for p in self.params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in self.params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        
    def __dataset(self, TrainProcessor, ValProcessor):
            
        # Combine the training inputs into a TensorDataset.

        dataset = TensorDataset(torch.tensor(TrainProcessor.input_ids[:15000]), 
                                torch.tensor(TrainProcessor.input_mask[:15000]), 
                                torch.tensor(TrainProcessor.label_ids[:15000]))
        

        val_dataset = TensorDataset(torch.tensor(ValProcessor.input_ids[:20000]),
                                    torch.tensor(ValProcessor.input_mask[:20000]), 
                                    torch.tensor(ValProcessor.label_ids[:20000]))

        train_dataloader = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = batch_size)
        validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)

        return train_dataloader, validation_dataloader
    
    def __new_dataset(self, DataProcessor):
        
        dataset = TensorDataset(torch.tensor(DataProcessor.input_ids[:15000]), 
                                torch.tensor(DataProcessor.input_mask[:15000]), 
                                torch.tensor(DataProcessor.label_ids[:15000]))

        val_dataset = TensorDataset(torch.tensor(DataProcessor.input_ids[15000:20000]), 
                                torch.tensor(DataProcessor.input_mask[15000:20000]), 
                                torch.tensor(DataProcessor.label_ids[15000:20000]))

        train_dataloader = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = batch_size)
        validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)

        return train_dataloader, validation_dataloader
    
    def format_time(self, elapsed):
        
        #Takes a time in seconds and returns a string hh:mm:ss
       
        elapsed_rounded = int(round((elapsed)))    
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):
        
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat[labels_flat!=0] == labels_flat[labels_flat!=0]) / len(labels_flat[labels_flat!=0])
    
    def train(self, epochs = epochs):
        
        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_stats = []
        total_t0 = time.time()

        for epoch_i in range(epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0

            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)
                    
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))

                # Unpack this training batch from our dataloader. 
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()        

                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                loss, logits = self.model(b_input_ids, 
                                     token_type_ids=None, #b_segment, 
                                     attention_mask=b_input_mask, 
                                     labels=b_labels)
                
                total_train_loss += loss.item()        
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()
            avg_train_loss = total_train_loss / len(self.train_dataloader)            

            training_time = self.format_time(time.time() - t0)

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in self.validation_dataloader:

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():        

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    (loss, logits) = self.model(b_input_ids, 
                                           token_type_ids=None, #b_segment, 
                                           attention_mask=b_input_mask,
                                           labels=b_labels)

                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)
            print(self.tokenizer.convert_ids_to_tokens(b_input_ids[0, :]))   
            last = np.argmax(logits, axis=2)
            part = last[0,:]
            part_true = label_ids[0,:]
            print("Last true = ", part_true)
            print("Last prediction", part)

            avg_val_accuracy = total_eval_accuracy / len(self.validation_dataloader)
            print("  Accuracy: {0:.3f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.validation_dataloader)

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.3f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

            print("  Average training loss: {0:.3f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))
            torch.save(self.model.state_dict(), "Chkpt2.pth")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))
        torch.save(self.model.state_dict(), "Chkpt2.pth")
'''
