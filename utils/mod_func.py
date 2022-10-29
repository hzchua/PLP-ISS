from transformers import AutoModel, BertTokenizerFast
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import pandas as pd
import torch, re, random
import numpy as np
import json
import pprint

# data = {"intents": [
# {"tag": "greetings",
#  "responses": ["Howdy!", "Hello", "How are you doing?",   "Greetings!", "How do you do?", "Hi", "Nice to meet you!!", "Pleasure to meet you!!"]},
# {"tag": "apologies",
#  "responses": ["I'm sorry to hear that", "Hope you feel better", "Tomorrow will be a better day!"]},
# {"tag": "age",
#  "responses": ["My age is a secret… hehe", "Don't tell you!!", "Forever 18!!", "Forever young!!"]},
# {"tag": "bot",
#  "responses": ["I am Sense-R. I am here to help you to find out more on restaurant related matters.", "Sense-R here to help you with restaurant matters"]},
# {"tag": "annoy",
#  "responses": ["I'm sorry if I offended you.", "Apologies for any offending comments I made"]},
# {"tag": "compliments",
#  "responses": ["Thank you for your compliments!!", "Thank you!!","Thanks for your praise!"]},
# {"tag": "help",
#  "responses": ["Sure, I will be glad to help!", "Sense-R is here to assist you!","Leave it to Sense-R to resolve your problem!"]},
# {"tag": "farewell",
#  "responses": ["Goodbye!!", "Cya soon!!","Catch you later!!", "Hope to see you around again!"]},
# {"tag": "okay",
#  "responses": ["Okay!", "Alright!","No problem!!", "No issue"]},
# {"tag": "wait",
#  "responses": ["Okay! I will wait for you."]},
# {"tag": "work",
#  "responses": ["I work for my developer.", "I'm here to service you.", "Created by my developer to service you :)"]},
# {"tag": "joke",
#  "responses": ["I take my work seriously. I don't like to joke", "I do not joke while working."]},
# {"tag": "busy",
#  "responses": ["I am fine, do you have anything to ask me?", "I am busy.... but I am still here to assist you :D"]},
# {"tag": "location",
#  "responses": ["I live and work in Singapore","You can find me mostly in computers or laptops now.", "I live inside your heart :)"]},
# {"tag": "repeat",
#  "responses": ["I'm sorry, can you repeat again?","I was unable to catch that, can you repeat again?", "Pardon me, can you repeat again?"]}
# ]}

# with open("intents.json","w") as f:
#     json.dump(data,f, indent=1)

f = open("data/intents.json")
data = json.load(f)
f.close()
total_labels = len(data["intents"])

class BERT_Arch(nn.Module):
    def __init__(self, bert, labels):      
        
       super(BERT_Arch, self).__init__()
       self.bert = bert 
       self.dropout = nn.Dropout(0.2)
       self.relu =  nn.ReLU()
       self.fc1 = nn.Linear(768,512)
       self.fc2 = nn.Linear(512,256)
       self.fc3 = nn.Linear(256,labels)
       self.softmax = nn.LogSoftmax(dim=1)
       
    def forward(self, sent_id, mask):
        #pass the inputs to the model  
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
        
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        x = self.softmax(x)
        
        return x

############# For conversational models ################################

def encoding(filepath="data/dataset.xlsx", sheet="conv"):
    df = pd.read_excel(filepath, sheet_name=sheet)
    label_e = LabelEncoder()
    df['Label'] = label_e.fit_transform(df['Label'])
    
    return label_e

def get_prediction(str, model, tokenizer, max_seq_len = 8, device="cpu"):
    le = encoding()
    
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
        # print("Intent Identified: ", le.inverse_transform(preds)[0])
        
        return le.inverse_transform(preds)[0]

def get_response(message, model): 
    loaded_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    intent = get_prediction(message,model, loaded_tokenizer)
    for i in data['intents']: 
        if i["tag"] == intent:
            result = random.choice(i["responses"])
            break
    # print(f"Response : {result}")
    # return "Intent: "+ intent + '\n' + "Response: " + result
    return result

############### For intent model #######################################

def int_encoding(filepath="data/dataset.xlsx", sheet="main"):
    df = pd.read_excel(filepath, sheet_name=sheet)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    return le, tokenizer

def get_intent(str, model, max_seq_len=8, device="cpu"):
    le, tokenizer = int_encoding()
    
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
        # print("Intent Identified: ", le.inverse_transform(preds)[0])
        
        return le.inverse_transform(preds)[0]