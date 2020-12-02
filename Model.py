import time
import datetime
import random
import warnings
import h5py
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertForTokenClassification, AutoModelWithLMHead
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from scipy.special import softmax
warnings.filterwarnings('ignore', category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))


class GetIndices:

    def __init__(self, ftype, data_dir):
        self.file_names = [data_dir + 'input_ids_' + ftype + '.txt', data_dir + 'input_mask_' + ftype + '.txt',
                           data_dir + 'label_ids_' + ftype + '.txt']
        self.file_hdf = data_dir + ftype
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
            self.input_ids = f['input_ids'][:, :]
            self.input_mask = f['input_mask'][:, :]
            self.label_ids = f['label_ids'][:, :]


class TsyaModel:

    def __init__(self, seed_val,  adam_options, tokenizer, label_list, from_rubert, multilingual=True,
                 config_of_model=None, weight_path=None, train_from_chk=False, device=device):
        if weight_path is not None:
            self.weight_path = weight_path
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(self.label_list)}
        if multilingual:
            self.model = BertForTokenClassification.from_pretrained(
                'bert-base-multilingual-cased',
                num_labels=len(self.label_list),
                output_attentions=False,
                output_hidden_states=False
            )
        else:
            self.model = BertForTokenClassification.from_pretrained(
                'bert-base-cased',
                num_labels=len(self.label_list),
                output_attentions=False,
                output_hidden_states=False
            )
        if train_from_chk:
            self.model.load_state_dict(torch.load(self.weight_path, map_location=torch.device('cpu')))

        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.model.parameters(),
                               **adam_options
                               )
        self.model.to(device)
        self.seed_val = seed_val  # 42

    def format_time(self, elapsed):

        # Takes a time in seconds and returns a string hh:mm:ss
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):

        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat[labels_flat != 0] == labels_flat[labels_flat != 0]) / len(labels_flat[labels_flat != 0])

    def _dataset(self, train_processor, val_processor, batch_size):

        dataset = TensorDataset(torch.from_numpy(train_processor.input_ids),
                                torch.from_numpy(train_processor.input_mask),
                                torch.from_numpy(train_processor.label_ids))

        val_dataset = TensorDataset(torch.from_numpy(val_processor.input_ids),
                                    torch.from_numpy(val_processor.input_mask),
                                    torch.from_numpy(val_processor.label_ids))

        train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
        validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

        return train_dataloader, validation_dataloader

    def train(self, chkp_path, train_data_processor, val_data_processor, epochs, batch_size, do_validation=False):
        # if not chkp_path:
        #     chkp_path = self.weight_path

        train_dataloader, validation_dataloader = self._dataset(train_data_processor, val_data_processor, batch_size)

        print("Dataloader is created")

        total_steps = len(train_dataloader) * epochs

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
            if do_validation:
                print("")
                print("Running Validation...")

                t0 = time.time()

                self.model.eval()

                total_eval_accuracy = 0
                total_eval_loss = 0
                b_input_ids = np.nan
                label_ids = np.nan
                logits = np.nan

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
                # with open("logs.txt", 'r+') as f:
                #     f.write("  Validation Loss: {0:.3f}".format(avg_val_loss))
                #     f.write("  Validation took: {:}".format(validation_time))
                #     f.write("  Average training loss: {0:.3f}".format(avg_train_loss))
                #     f.write("  Training epoch took: {:}".format(training_time))
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

    def predict_batch(self, prediction_dataloader, nopad):
        self.model.eval()
        predicts_full = []
        probability = []
        step = 0
        probs_o = np.nan
        probs = np.nan
        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():

                output = self.model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask)
            logits = output[0].detach().cpu().numpy()
            # norm = np.linalg.norm(logits, axis=2)
            prediction = np.argmax(logits, axis=2)
            o_index = self.label_list.index('O')
            predicts = []
            for i in range(len(b_input_ids)):

                # probs = np.divide(np.argmax(logits, axis=2)[i][:nopad[step]], norm[i][:nopad[step]])
                soft = softmax(logits, axis=2)
                probs = np.amax(soft, axis=2)[i][:nopad[step]]
                probs_o = [soft[i][el][o_index] for el in range(nopad[step])]
                predicts.append(prediction[i, :nopad[step]])
                step += 1
            probability.append(probs)
            predicts_full.append(predicts)

        return predicts_full, probs, probs_o
