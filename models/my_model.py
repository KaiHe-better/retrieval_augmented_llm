# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class My_Model(nn.Module):
    def __init__(self, args, LLM, LLM_tokenizer,  stop_token_ids):
        nn.Module.__init__(self)
        self.args = args
        self.stop_token_ids = stop_token_ids
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0


        self.LLM = LLM
        self.LLM_tokenizer = LLM_tokenizer


    
    def forward(self, input_ids, attention_mask, retriever_txt=None):
        # if self.args.if_RA:
        #     pass
        
        # else:

        #     with torch.no_grad():
        generation = self.generate(input_ids, attention_mask, min(self.args.max_new_tokens, self.args.max_length-input_ids.size(1)) )
            
          
        return generation


    def generate(self, input_ids, attention_mask, max_tokens):
        if max_tokens == 0:
            self.prompt_exceed_max_length += 1
            self.args.print_logger.warning("Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            self.args.print_logger.warning("The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")


        outputs = self.LLM.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=True, temperature=self.args.temperature, top_p=self.args.top_p, 
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            eos_token_id=self.stop_token_ids
        )

        generation = self.LLM_tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)
        return generation