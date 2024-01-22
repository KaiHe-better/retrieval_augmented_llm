import json
import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os




class MedMCQA(Dataset):

    def __init__(self, args, data_file, LLM_tokenizer):
        self.args = args
        self.LLM_tokenizer = LLM_tokenizer
        self.map_dic = {"1": "A", "2": "B", "3": "C", "4": "D"}
        with open(data_file, "r") as f:
            self.data = f.readlines()
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        data_item = json.loads(self.data[index])  
        question = data_item['question']
        answ = self.map_dic[str(data_item['cop'])]
        
        label = [self.LLM_tokenizer._convert_token_to_id(answ)]
        # label = self.LLM_tokenizer._convert_token_to_id(answ)
        one_hot_label = torch.zeros(self.LLM_tokenizer.vocab_size)
        one_hot_label.index_fill_(0, torch.tensor(label), torch.tensor(1))

        options = ' '
        for k, v in zip(["A", "B", "C", "D"], ['opa', 'opb', 'opc', 'opd']):
            options += "<"+str(k) + "> " + str(data_item[v])+ ". " 

        return {"question": question, 'options': options,  'label': label, "answer": answ,  "one_hot_label": one_hot_label}



def collate_fn_MedMCQA(data):
    batch_data = {'question': [],  'options': [], 'label': [], "answer":[], "one_hot_label":[] }
    for data_item in data:
        for k, v in batch_data.items():
            tmp = data_item[k]
            batch_data[k].append(tmp)
            
    batch_data['question'] = batch_data['question']
    batch_data['answer']   = batch_data['answer']
    batch_data['options']  = batch_data['options']  
    batch_data['label']    = batch_data['label']
    batch_data['one_hot_label']   = torch.stack(batch_data['one_hot_label'])

    return batch_data

 

def get_loader_MedMCQA(args, triever_tokenizer, train_file_path, dev_file_path, test_file_path) :
    
    train_dataset = MedMCQA(args, train_file_path, triever_tokenizer)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn_MedMCQA,
                                  )       

    # for demonstration
    dev_dataset = MedMCQA(args, dev_file_path, triever_tokenizer)
    dev_data_loader = DataLoader(dataset=dev_dataset,
                                 batch_size=args.demons_cnt,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn_MedMCQA,
                                ) 
    

    test_dataset = MedMCQA(args, test_file_path, triever_tokenizer)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_MedMCQA,
                                 )    

    return train_data_loader, dev_data_loader, test_data_loader, args



