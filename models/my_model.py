# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, MultiheadAttention, Linear, Dropout, LayerNorm
import torch.nn.functional as F
from utils.utils import combine_doc
from sklearn.metrics import  accuracy_score
from sklearn.cluster import KMeans

class My_MI_learner(nn.Module):
    def __init__(self, args, vocab_size):
        nn.Module.__init__(self)
        self.args = args

        factory_kwargs = {'device': self.args.device}
        self.trans_ques = TransformerEncoderLayer(self.args.d_model, self.args.nhead, dropout=self.args.dropout, batch_first=True, **factory_kwargs)
        self.trans_doc = TransformerEncoderLayer(self.args.d_model, self.args.nhead, dropout=self.args.dropout, batch_first=True, **factory_kwargs)

        self.multi_head_ques = My_Multi_Head_Attn(args, factory_kwargs=factory_kwargs)
        self.multi_head_doc = My_Multi_Head_Attn(args, factory_kwargs=factory_kwargs)

        self.linear_mse = Linear(self.args.d_model, 1, **factory_kwargs)
        self.linear_kl = Linear(self.args.d_model, vocab_size, **factory_kwargs)

        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, query_emb, ques_att_masks, bags_list, batch_logit_log_softmax, batch_loss, retriever, triever_tokenizer, train_flag):
        total_mse_logit = []
        total_kl_logit = []
        select_doc = []
        select_doc_num = []
        for bag, raw_ques_emb, ques_att_mask in zip(bags_list, query_emb, ques_att_masks):
            with torch.no_grad():
                doc_input = triever_tokenizer(bag, truncation=True, return_tensors='pt', max_length=self.args.chunk_size, padding=True).to(self.args.device)
                raw_doc_emb = retriever(**doc_input).last_hidden_state

            ques_att_mask =(1- ques_att_mask).bool() 
            doc_att_mask = (1- doc_input['attention_mask']).bool()
            raw_ques_emb = self.trans_ques(raw_ques_emb, src_key_padding_mask=ques_att_mask)[0, :].unsqueeze(0).unsqueeze(0)
            raw_doc_emb  = self.trans_doc(raw_doc_emb, src_key_padding_mask=doc_att_mask)[:, 0, :].unsqueeze(0)
            
            que_emb, att_weights  = self.multi_head_ques(raw_ques_emb, raw_doc_emb)
            doc_emb, _  = self.multi_head_doc(raw_doc_emb, que_emb)

            # select_index = torch.where( att_weights.squeeze() >= torch.quantile(att_weights, self.args.quantile_num) )[0]
            select_index = torch.where( att_weights.squeeze() >= 1/len(bag)* self.args.quantile_num )[0]
            select_doc.append( combine_doc([bag[i] for i in select_index ]) )
            select_doc_num.append(len(select_index))
            
            if train_flag:
                doc_emb = torch.mean(doc_emb, dim=1).squeeze()

                if "mse" in self.args.loss_list:
                    mse_logit = self.linear_mse(doc_emb) 
                    total_mse_logit.append(mse_logit)

                if "kl" in self.args.loss_list:
                    kl_logit = self.linear_kl(doc_emb) 
                    MI_logit_log_softmax = F.log_softmax(kl_logit, dim=0)
                    total_kl_logit.append(MI_logit_log_softmax)

        total_loss = 0
        if train_flag:
            if "mse" in self.args.loss_list:
                MSL_loss = self.mse_loss( torch.stack(total_mse_logit).squeeze(), 1/batch_loss.float() )
                total_loss+=MSL_loss
            if "kl" in self.args.loss_list:
                KLL_loss = self.kl_loss(torch.stack(total_kl_logit).squeeze(), batch_logit_log_softmax) 
                total_loss+=KLL_loss

        return total_loss, select_doc, select_doc_num
    





class My_Multi_Head_Attn(nn.Module):
    def __init__(self, args, factory_kwargs):
        nn.Module.__init__(self)
        self.args = args
        self.multi_attn = MultiheadAttention(self.args.d_model, self.args.nhead, dropout=self.args.dropout, batch_first=True, **factory_kwargs)

        self.linear1 = Linear(self.args.d_model, self.args.dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(self.args.dim_feedforward, self.args.d_model, **factory_kwargs)

        self.norm1 = LayerNorm(self.args.d_model, eps=self.args.layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(self.args.d_model, eps=self.args.layer_norm_eps, **factory_kwargs)
        self.dropout = Dropout(self.args.dropout)
        self.dropout1 = Dropout(self.args.dropout)
        self.dropout2 = Dropout(self.args.dropout)
        self.activation = F.gelu

    def forward(self, q, k_v, attn_mask=None, key_padding_mask= None):
        tmp_att, att_weights = self._sa_block(q, k_v, k_v,  attn_mask=attn_mask, key_padding_mask= key_padding_mask)
        q = self.norm1(q + tmp_att)
        q = self.norm2(q + self._ff_block(q))
        return  q, att_weights
    
    def _sa_block(self, q, k, v, attn_mask=None, key_padding_mask=None) :
        attn_output, attn_output_weights = self.multi_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, 
                                                          need_weights=True, average_attn_weights=True)

        return self.dropout1(attn_output), attn_output_weights

    def _ff_block(self, x) :
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

