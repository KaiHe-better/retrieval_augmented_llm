# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class My_Model(nn.Module):
    def __init__(self, LLM, LLM_tokenizer, args,  stop_token_ids):
        nn.Module.__init__(self)
        self.args = args
        self.stop_token_ids = stop_token_ids

        self.LLM = LLM
        self.LLM_tokenizer = LLM_tokenizer


    
    def forward(self, my_input):
        my_input = self.LLM_tokenizer(my_input, return_tensors='pt')
        outputs = self.LLM.generate(
            **my_input,
            do_sample=True, temperature=self.args.temperature, top_p=self.args.top_p, 
            max_new_tokens=self.args.max_tokens,
            num_return_sequences=1,
            eos_token_id=self.stop_token_ids
        )

        generation = self.LLM_tokenizer.decode(outputs[0][my_input["inpit_ids"].size(1):], skip_special_tokens=True)
            
          
        return generation

