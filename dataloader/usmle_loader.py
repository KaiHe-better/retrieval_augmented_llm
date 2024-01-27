import json
import numpy as np
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os



class USMLE(Dataset):

    def __init__(self, args, data_file, LLM_tokenizer, rewrite_file_path):
        self.args = args
        self.LLM_tokenizer = LLM_tokenizer

        with open(data_file, "r") as f:
            self.data = f.readlines()

        with open(rewrite_file_path, "r") as f:
            self.rewrte_data = eval(f.readlines()[0])

        # assert len(self.data) == len(self.rewrte_data)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        data_item = eval(self.data[index])
        rewrte_data_item = self.rewrte_data[index][:self.args.rewrite_num]

        question = data_item['question'].replace("\n\n", "\n")
        answ = data_item['answer_idx']
        
        label = [self.LLM_tokenizer._convert_token_to_id(answ)]
        # label = self.LLM_tokenizer._convert_token_to_id(answ)
        one_hot_label = torch.zeros(self.LLM_tokenizer.vocab_size)
        one_hot_label.index_fill_(0, torch.tensor(label), torch.tensor(1))

        options = ' '
        for k, v in data_item['options'].items():
            options += "<"+str(k) + "> " + str(v)+ ". " 

        return {"question": question, 'options': options,  'label': label, "answer": answ,  "one_hot_label": one_hot_label, "rewrite_question": rewrte_data_item} 



def collate_fn_USMLE(data):
    batch_data = {'question': [],  'options': [], 'label': [], "answer":[], "one_hot_label":[], "rewrite_question":[]}
    for data_item in data:
        for k, v in batch_data.items():
            tmp = data_item[k]
            batch_data[k].append(tmp)
            
    batch_data['question'] = batch_data['question']
    batch_data['rewrite_question'] = batch_data['rewrite_question']
    batch_data['answer']   = batch_data['answer']
    batch_data['options']  = batch_data['options']  
    batch_data['label']    = batch_data['label']
    batch_data['one_hot_label']   = torch.stack(batch_data['one_hot_label'])

    return batch_data

 

def get_loader_USMLE(args, triever_tokenizer, train_file_path, dev_file_path, test_file_path, rewrite_train_file_path, rewrite_dev_file_path, rewrite_test_file_path) :
    
    train_dataset = USMLE(args, train_file_path, triever_tokenizer, rewrite_train_file_path)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn_USMLE,
                                  )       

    # for demonstration
    dev_dataset = USMLE(args, dev_file_path, triever_tokenizer, rewrite_dev_file_path)
    dev_data_loader = DataLoader(dataset=dev_dataset,
                                 batch_size=args.demons_cnt,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn_USMLE,
                                ) 
    

    test_dataset = USMLE(args, test_file_path, triever_tokenizer, rewrite_test_file_path)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_USMLE,
                                 )    

    return train_data_loader, dev_data_loader, test_data_loader, args



