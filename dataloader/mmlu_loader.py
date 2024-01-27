import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

class MMLU(Dataset):

    def __init__(self, args, data_file, LLM_tokenizer, rewrite_file_path):
        self.args = args
        self.LLM_tokenizer = LLM_tokenizer

        with open(data_file, "r") as f:
            self.data = f.readlines()

        # with open(rewrite_file_path, "r") as f:
        #     self.rewrte_data = eval(f.readlines()[0])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        data_item = self.data[index].split(",")
        question = data_item[0]
        answ = data_item[-1].replace("\n", "")
        
        label = [self.LLM_tokenizer._convert_token_to_id(answ)]
        one_hot_label = torch.zeros(self.LLM_tokenizer.vocab_size)
        one_hot_label.index_fill_(0, torch.tensor(label), torch.tensor(1))

        options = ' '
        for k, v in zip(["A", "B", "C", "D"], data_item[1:5]):
            options += "<"+str(k) + "> " + str(v)+ " " 

        return {"question": question, 'options': options,  'label': label, "answer": answ,  "one_hot_label": one_hot_label}



def collate_fn_MMLU(data):
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

 

def get_loader_MMLU(args, triever_tokenizer, test_file_path, rewrite_train_file_path) :

    test_dataset = MMLU(args, test_file_path, triever_tokenizer, rewrite_train_file_path)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_MMLU,
                                 )    

    return "train_data_loader", "dev_data_loader", test_data_loader, args



