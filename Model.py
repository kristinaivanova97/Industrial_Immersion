import json
import time
import datetime
import random
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import h5py
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertTokenizer, BertForTokenClassification, AutoModelWithLMHead, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: {}".format(device))


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
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        # if not weight_path:
        #     self.weight_path = weight_path
        self.label_list = config['label_list']
        config['num_labels'] = len(config['label_list'])
        self.label_map = {label: i for i, label in enumerate(self.label_list)}
        if config['from_bert']:

            self.tokenizer = BertTokenizer.from_pretrained(**config['config_of_tokenizer'])

            self.model = BertForTokenClassification.from_pretrained(
                **config['config_of_model']
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(**config['config_of_tokenizer'])

            self.model = AutoModelWithLMHead.from_pretrained(
                **config['config_of_model']
            )

        if train_from_chk:
            self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))


        self.optimizer = AdamW(self.model.parameters(),
                          **config['adam_options']
                         )
        self.model.to(device)
        self.seed_val = config['seed_val'] #42
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']

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

        dataset = TensorDataset(torch.from_numpy(train_processor.input_ids),
                                torch.from_numpy(train_processor.input_mask),
                                torch.from_numpy(train_processor.label_ids))

        val_dataset = TensorDataset(torch.from_numpy(val_processor.input_ids),
                                    torch.from_numpy(val_processor.input_mask),
                                    torch.from_numpy(val_processor.label_ids))

        train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=self.batch_size)
        validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=self.batch_size)

        return train_dataloader, validation_dataloader


    def train(self, chkp_path, train_data_processor, val_data_processor):
        # if not chkp_path:
        #     chkp_path = self.weight_path

        train_dataloader, validation_dataloader = self._dataset(train_data_processor, val_data_processor)

        print("Dataloader is created")

        total_steps = len(train_dataloader) * self.epochs

        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=total_steps)

        params = list(self.model.named_parameters())

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

        training_stats = []
        total_t0 = time.time()

        for epoch_i in range(self.epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0

            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader),
                                                                                elapsed))

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                self.model.zero_grad()

                loss, logits = self.model(b_input_ids,
                                          token_type_ids=None,  # b_segment,
                                          attention_mask=b_input_mask,
                                          labels=b_labels.to(dtype=torch.long))

                total_train_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)

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
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    (loss, logits) = self.model(b_input_ids,
                                                token_type_ids=None,  # b_segment,
                                                attention_mask=b_input_mask,
                                                labels=b_labels.to(dtype=torch.long))

                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                total_eval_accuracy += self.flat_accuracy(logits, label_ids)
            print(self.tokenizer.convert_ids_to_tokens(b_input_ids[0, :]))
            last = np.argmax(logits, axis=2)
            part = last[0, :]
            part_true = label_ids[0, :]
            print("Last true = ", part_true)
            print("Last prediction", part)

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.3f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

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
