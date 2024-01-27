import json
import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import os


class OTTQA(Dataset):

    def __init__(self, args, data_file, triever_tokenizer):
        self.args = args
        self.triever_tokenizer = triever_tokenizer
        
        with open(data_file, 'r') as file:
            self.data = json.load(file)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        data_item = self.data[index]
        question = data_item['question'].replace("\n\n", "\n")
        
        if 'answer-text' in data_item.keys():
            answ = data_item['answer-text']
        else:
            answ = " "

        label = self.triever_tokenizer(answ, add_special_tokens=False)["input_ids"]
        one_hot_label = torch.zeros(self.triever_tokenizer.vocab_size)
        one_hot_label.index_fill_(0, torch.tensor(label), torch.tensor(1))
        return {"question": question,  "answer": answ, "label": label, "one_hot_label": one_hot_label}



def collate_fn_OTTQA(data):
    batch_data = {'question': [], "answer":[],  "label":[], "one_hot_label": [] }
    for data_item in data:
        for k, v in batch_data.items():
            tmp = data_item[k]
            batch_data[k].append(tmp)
            
    batch_data['question'] = batch_data['question']
    batch_data['answer']   = batch_data['answer']
    batch_data['label']   = batch_data['label']
    batch_data['one_hot_label']   = torch.stack(batch_data['one_hot_label'])
    
    return batch_data

 

def get_loader_OTTQA(args, triever_tokenizer, train_file_path, dev_file_path, test_file_path) :
    
    train_dataset = OTTQA(args, train_file_path, triever_tokenizer)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn_OTTQA,
                                  )       

    # for demonstration
    dev_dataset = OTTQA(args, dev_file_path, triever_tokenizer)
    dev_data_loader = DataLoader(dataset=dev_dataset,
                                 shuffle=False,
                                 pin_memory=True,
                                 batch_size=args.test_batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn_OTTQA,
                                ) 
    

    test_dataset = OTTQA(args, test_file_path, triever_tokenizer)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_OTTQA,
                                 )    

    return train_data_loader, dev_data_loader, test_data_loader, args



