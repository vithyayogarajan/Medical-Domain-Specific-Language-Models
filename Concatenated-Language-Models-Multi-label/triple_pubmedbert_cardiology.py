import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
from sklearn.metrics import classification_report
import torch
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

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

df = pd.read_csv("cardioMIMICthirddischargeallwithlabels.csv")

df['list'] = df[df.columns[4:]].values.tolist()
new_df = df[['text3','text2','text1','list']].copy()
new_df.columns = ['text3','text2','text1','list']

del df

# Defining some key variables that will be used later on in the training
MAX_LEN1 = 512
MAX_LEN2 = 512
MAX_LEN3 = 512
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
EPOCHS = 30
LEARNING_RATE = 7e-06
tokenizer =  AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')


class Dataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text1 = dataframe.text1
        self.text2 = dataframe.text2
        self.text3 = dataframe.text3
        self.targets = self.data.list
        self.max_len1 = MAX_LEN1
        self.max_len2 = MAX_LEN2
        self.max_len3 = MAX_LEN3

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, index):
        text1 = str(self.text1[index])
        text1 = " ".join(text1.split())
        text2 = str(self.text2[index])
        text2 = " ".join(text2.split())
        text3 = str(self.text3[index])
        text3 = " ".join(text3.split())

        inputs_text1 = self.tokenizer.encode_plus(
            text1,
            None,
            add_special_tokens=True,
            max_length=self.max_len1,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids_text1 = inputs_text1['input_ids']
        mask_text1 = inputs_text1['attention_mask']
        token_type_ids_text1 = inputs_text1["token_type_ids"]
        
        
        inputs_text2 = self.tokenizer.encode_plus(
            text2,
            None,
            add_special_tokens=True,
            max_length=self.max_len2,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids_text2 = inputs_text2['input_ids']
        mask_text2 = inputs_text2['attention_mask']
        token_type_ids_text2 = inputs_text2["token_type_ids"]
        
        inputs_text3 = self.tokenizer.encode_plus(
            text3,
            None,
            add_special_tokens=True,
            max_length=self.max_len3,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids_text3 = inputs_text3['input_ids']
        mask_text3 = inputs_text3['attention_mask']
        token_type_ids_text3 = inputs_text3["token_type_ids"]

        return {
            'ids_text1': torch.tensor(ids_text1, dtype=torch.long),
            'mask_text1': torch.tensor(mask_text1, dtype=torch.long),
            'token_type_ids_text1': torch.tensor(token_type_ids_text1, dtype=torch.long),
            'ids_text2': torch.tensor(ids_text2, dtype=torch.long),
            'mask_text2': torch.tensor(mask_text2, dtype=torch.long),
            'token_type_ids_text2': torch.tensor(token_type_ids_text2, dtype=torch.long),
            'ids_text3': torch.tensor(ids_text3, dtype=torch.long),
            'mask_text3': torch.tensor(mask_text3, dtype=torch.long),
            'token_type_ids_text3': torch.tensor(token_type_ids_text3, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


train_size = 0.8
train_dataset=new_df.sample(frac=train_size,random_state=200)
test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))



training_set = Dataset(train_dataset, tokenizer, MAX_LEN1)
testing_set = Dataset(test_dataset, tokenizer, MAX_LEN1)


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


class TripleBERTClass(torch.nn.Module):
    def __init__(self):
        super(TripleBERTClass, self).__init__()
        self.text1 = transformers.AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.text2 = transformers.AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.text3 = transformers.AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.l2 = torch.nn.Dropout(0.3)
        self.avgpool = torch.nn.AvgPool1d(2, 2)
        self.l3 = torch.nn.Linear(1152, 30) ## 30 is the number of labels
    
    def forward(self, ids_text1, mask_text1, ids_text2, mask_text2, ids_text3, mask_text3):
        _, output_text1= self.text1(ids_text1, attention_mask = mask_text1)
        _, output_text2= self.text2(ids_text2, attention_mask = mask_text2)
        _, output_text3= self.text3(ids_text3, attention_mask = mask_text3)
        
        text1_features = output_text1.unsqueeze(1)
        text1_features_pooled = self.avgpool(text1_features)
        text1_features_pooled = text1_features_pooled.squeeze(1)
        
        text2_features = output_text2.unsqueeze(1)
        text2_features_pooled = self.avgpool(text2_features)
        text2_features_pooled = text2_features_pooled.squeeze(1)

        text3_features = output_text3.unsqueeze(1)
        text3_features_pooled = self.avgpool(text3_features)
        text3_features_pooled = text3_features_pooled.squeeze(1)
        
        combined_features = torch.cat((
            text1_features_pooled, 
            text2_features_pooled,
            text3_features_pooled), 
            dim=1
        )
                
        output_2 = self.l2(combined_features)
#         print(output_2.size())
        output = self.l3(output_2)
        return output

model = TripleBERTClass()
model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids_text1 = data['ids_text1'].to(device, dtype = torch.long)
        mask_text1 = data['mask_text1'].to(device, dtype = torch.long)
        token_type_ids_text1 = data['token_type_ids_text1'].to(device, dtype = torch.long)
        ids_text2 = data['ids_text2'].to(device, dtype = torch.long)
        mask_text2 = data['mask_text2'].to(device, dtype = torch.long)
        token_type_ids_text2 = data['token_type_ids_text2'].to(device, dtype = torch.long) 
        ids_text3 = data['ids_text3'].to(device, dtype = torch.long)
        mask_text3 = data['mask_text3'].to(device, dtype = torch.long)
        token_type_ids_text3 = data['token_type_ids_text3'].to(device, dtype = torch.long) 
        targets = data['targets'].to(device)

        optimizer.zero_grad()
        
         # forward pass
        outputs = model(
            ids_text1, 
            mask_text1,
            ids_text2,
            mask_text2,
            ids_text3,
            mask_text3,
        )
        
       
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
            ids_text1 = data['ids_text1'].to(device, dtype = torch.long)
            mask_text1 = data['mask_text1'].to(device, dtype = torch.long)
            ids_text2 = data['ids_text2'].to(device, dtype = torch.long)
            mask_text2 = data['mask_text2'].to(device, dtype = torch.long)
            ids_text3 = data['ids_text3'].to(device, dtype = torch.long)
            mask_text3 = data['mask_text3'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids_text1, mask_text1,ids_text2, mask_text2, ids_text3, mask_text3)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


f1_score_micro = []
f1_score_macro = []
myfile = open('fscore_eachepoch_triplepubmedbert_cardio_output.txt', 'w')
for epoch in range(EPOCHS):
    train(epoch)
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_micro = metrics.f1_score(targets, outputs, average='micro')	
    f1_macro = metrics.f1_score(targets, outputs, average='macro')
    f1_score_micro.append(f1_micro)
    f1_score_macro.append(f1_macro)
    myfile.write(f"Epochs = {epoch}\n")
    myfile.write("This is the output of cardio, triple PubMedBERT for discharge summary 1/3, 3*512 \n")
    myfile.write(classification_report(targets, outputs))
    myfile.write("\n")

myfile.write(f"F1 Score (Micro) = {f1_score_micro}\n")
myfile.write(f"F1 Score (Macro) = {f1_score_macro}\n")

myfile.close()


print(classification_report(targets, outputs))






