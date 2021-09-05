# Importing stock ml libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
import time
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report
)

# # Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

df = pd.read_csv("cardio_mimic.csv")

df['list'] = df[df.columns[2:]].values.tolist() ### multi-label requires the labels to be in a list
new_df = df[['text', 'list']].copy()
new_df.columns = ['TEXT', 'list']

del df

MAX_LEN = 512 ### max sequence length for RoBERTa is 512
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
EPOCHS = 30
LEARNING_RATE = 9e-06
tokenizer =  AutoTokenizer.from_pretrained('allenai/biomed_roberta_base', truncation=True)

class Dataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.TEXT = dataframe.TEXT
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.TEXT)

    def __getitem__(self, index):
        TEXT = str(self.TEXT[index])
        TEXT = " ".join(TEXT.split())

        inputs = self.tokenizer.encode_plus(
            TEXT,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_dataset=new_df.sample(frac=train_size,random_state=200)
test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = Dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = Dataset(test_dataset, tokenizer, MAX_LEN)



train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)



class RoBERTaClass(torch.nn.Module):
    def __init__(self):
        super(RoBERTaClass, self).__init__()
        self.l1 = transformers.AutoModel.from_pretrained('allenai/biomed_roberta_base')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 30) ### 30 is the number of labels in cardio
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = RoBERTaClass()
model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

t0 = time.time()
f1_score_micro = []
f1_score_macro = []
myfile = open('fscore_eachepoch_biomedroberto_cardio_30labels_mimic.txt', 'w')
for epoch in range(EPOCHS):
    train(epoch)
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_micro = metrics.f1_score(targets, outputs, average='micro')	
    f1_macro = metrics.f1_score(targets, outputs, average='macro')
    f1_score_micro.append(f1_micro)
    f1_score_macro.append(f1_macro)
    print(f"epoch = {epoch}")
    print('{} seconds'.format(time.time() - t0))
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    myfile.write('{} seconds'.format(time.time() - t0))
    myfile.write("\n")
    myfile.write("This is the output of discharge summary, MIMIC-III, cardiovascular disease, with 30 labels, using BioMed-RoBERTa \n")
    myfile.write(classification_report(targets, outputs))
    myfile.write("\n")

myfile.write(f"F1 Score (Micro) = {f1_score_micro}\n")
myfile.write(f"F1 Score (Macro) = {f1_score_macro}\n")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")
print('{} seconds'.format(time.time() - t0))

myfile.close()


print(classification_report(targets, outputs))


with open('output_roberto_cardio.txt', 'w') as f:
        print("This is the output of discharge summary, MIMIC-III, cardiovascular disease, with 30 labels, using BioMed-RoBERTa",file=f)
        print("   ",file=f)
        print('classification report', classification_report(targets, outputs),file=f)


