import numpy as np
import pandas as pd
import re, torch, random, json, pprint
import torch.nn as nn

from utils.mod_func import BERT_Arch
from transformers import AutoModel, BertTokenizerFast
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torchinfo import summary
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import lr_scheduler, AdamW
torch.manual_seed(1234)

############## Parameters ###################################
device = "cpu"
df = pd.read_excel("dataset.xlsx", sheet_name="main")
total_labels = len(df["label"].unique())

model_path = "../models/int_model.pt"
le = LabelEncoder()
max_seq_len = 8
df['label'] = le.fit_transform(df['label'])
#############################################################

train_text, train_labels = df["text"], df["label"]
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

tokens_train = tokenizer(
    train_text.tolist(),
    max_length = max_seq_len,
    padding=True,
    truncation=True,
    return_token_type_ids=False
)

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

#define a batch size
batch_size = 16
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
# freeze all the parameters. This will prevent updating of model weights during fine-tuning.
for param in bert_model.parameters():
    param.requires_grad = False

model = BERT_Arch(bert_model, total_labels)
model = model.to(device)
summary(model)

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)
#compute the class weights
class_wts = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
print(class_wts)
# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)
# loss function
cross_entropy = nn.NLLLoss(weight=weights) 

# empty lists to store training and validation loss of each epoch
train_losses=[]
# number of training epochs
epochs = 200
# We can also use learning rate scheduler to achieve better results
lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

################ Model Training ###########
def train():

    model.train()
    total_loss = 0
    total_preds=[]
  
    for step,batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step,    len(train_dataloader)))

        batch = [r.to(device) for r in batch] 
        sent_id, mask, labels = batch
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        
        total_loss = total_loss + loss.item()
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # We are not using learning rate scheduler as of now
        # lr_sch.step()
        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()
        total_preds.append(preds)
        
    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
    
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

#############################################

for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train()
    
    # append training and validation loss
    train_losses.append(train_loss)
    # it can make your experiment reproducible, similar to set  random seed to all options where there needs a random seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f'\nTraining Loss: {sum(train_losses)/len(train_losses):.3f}')

torch.save(model, model_path)

##################### Start Process ###############

def get_intent(str, model, tokenizer):
    str = re.sub(r"[^a-zA-Z ]+", "", str)
    test_text = [str]
    model.eval()
 
    tokens_test_data = tokenizer(test_text,
                                 max_length = max_seq_len,
                                 padding=True,
                                 truncation=True,
                                 return_token_type_ids=False)
    
    test_seq = torch.tensor(tokens_test_data["input_ids"])
    test_mask = torch.tensor(tokens_test_data["attention_mask"])
 
    preds = None
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis = 1)
        print("Intent Identified: ", le.inverse_transform(preds)[0])
        
        return le.inverse_transform(preds)[0]


if __name__ == "__main__":
    loaded_model = torch.load(model_path)
    loaded_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    get_intent("Good morning!",loaded_model, loaded_tokenizer)
    get_intent("Give me top few restaurants",loaded_model, loaded_tokenizer)
    get_intent("Restaurant improvements?",loaded_model, loaded_tokenizer)