import json
import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import os



class HEADQA(Dataset):

    def __init__(self, args, data_file, LLM_tokenizer):
        self.args = args
        self.LLM_tokenizer = LLM_tokenizer
        self.map_dic = {"0": "A", "1": "B", "2": "C", "3": "D", "4":"E"}

        with open(data_file, "r") as f:
            self.data =  eval(f.readlines()[0])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        data_item = self.data[index]
        question = data_item['question']
        answ = self.map_dic[str(data_item['answer_id'])]
        label = [self.LLM_tokenizer._convert_token_to_id(answ)]
        one_hot_label = torch.zeros(self.LLM_tokenizer.vocab_size)
        one_hot_label.index_fill_(0, torch.tensor(label), torch.tensor(1))

        options = ' '
        for k, v in zip(["A", "B", "C", "D"], data_item['options']):
            options += "<"+str(k) + "> " + str(v)+ " " 

        return {"question": question, 'options': options,  'label': label, "answer": answ,  "one_hot_label": one_hot_label}



def collate_fn_HEADQA(data):
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

 

def get_loader_HEADQA(args, triever_tokenizer, train_file_path, dev_file_path, test_file_path) :
    
    train_dataset = HEADQA(args, train_file_path, triever_tokenizer)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn_HEADQA,
                                  )       

    # for demonstration
    dev_dataset = HEADQA(args, dev_file_path, triever_tokenizer)
    dev_data_loader = DataLoader(dataset=dev_dataset,
                                 batch_size=args.demons_cnt,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn_HEADQA,
                                ) 
    

    test_dataset = HEADQA(args, test_file_path, triever_tokenizer)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_HEADQA,
                                 )    

    return train_data_loader, dev_data_loader, test_data_loader, args



