import json
import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os



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
        answ = data_item['answer_idx']
        label = self.idx2label[answ]

        options = ' '
        for k, v in data_item['options'].items():
            options += "<"+str(k) + "> " + str(v)+ ". " 

        return {"question": data_item['question'], 'options': options,  'label': label, "answer": answ}



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
    demons_cnt = args.demons_cnt
    indices = torch.randperm(len(train_dataset)).tolist()
    sampler = SubsetRandomSampler(indices[:demons_cnt])
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size,
                                #    shuffle=True,
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn_USMLE,
                                   sampler=sampler,
                                  )       

    dev_dataset = USMLE(args, dev_file_path)
    dev_data_loader = DataLoader(dataset=dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn_USMLE,
                                ) 
    

    test_dataset = USMLE(args, test_file_path)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_USMLE,
                                 )    

    return train_data_loader, dev_data_loader, test_data_loader, args



