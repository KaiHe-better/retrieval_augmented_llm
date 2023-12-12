import json
import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os

class Prompt_Dataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class USMLE(Dataset):

    def __init__(self, args, data_file):
        self.args = args
        self.idx2label = {"A":0, "B":1, "C":2, "D":3}

        with open(data_file, "r") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        data_item = eval(self.data[index])
        question = data_item['question'].replace("\n\n", "\n")
        answ = data_item['answer_idx']
        label = self.idx2label[answ]

        options = ' '
        for k, v in data_item['options'].items():
            options += "<"+str(k) + "> " + str(v)+ ". " 

        return {"question": question, 'options': options,  'label': label, "answer": answ}



def collate_fn_USMLE(data):
    batch_data = {'question': [],  'options': [], 'label': [], "answer":[] }
    for data_item in data:
        for k, v in batch_data.items():
            tmp = data_item[k]
            batch_data[k].append(tmp)
            
    batch_data['question'] = batch_data['question']
    batch_data['options']  = batch_data['options']  
    batch_data['label']    = batch_data['label']
    batch_data['answer']   = batch_data['answer']
    
    return batch_data

 

def get_loader_USMLE(args, train_file_path, dev_file_path, test_file_path) :
    
    train_dataset = USMLE(args, train_file_path)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn_USMLE,
                                   
                                  )       

    # for demonstration
    dev_dataset = USMLE(args, dev_file_path)
    dev_data_loader = DataLoader(dataset=dev_dataset,
                                 batch_size=args.demons_cnt,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn_USMLE,
                                ) 
    

    test_dataset = USMLE(args, test_file_path)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_USMLE,
                                 )    

    return train_data_loader, dev_data_loader, test_data_loader, args



