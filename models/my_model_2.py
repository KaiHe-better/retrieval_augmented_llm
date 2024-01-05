# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, MultiheadAttention, Linear, Dropout, LayerNorm
import torch.nn.functional as F
from utils.utils import combine_doc
from sklearn.metrics import  accuracy_score
from sklearn.cluster import KMeans

class My_MI_learner(nn.Module):
    def __init__(self, args ):
        nn.Module.__init__(self)
        self.args = args

        factory_kwargs = {'device': self.args.device}
        self.trans_ques = TransformerEncoderLayer(self.args.d_model, self.args.nhead, dropout=0.1, batch_first=True, **factory_kwargs)
        self.trans_doc = TransformerEncoderLayer(self.args.d_model, self.args.nhead, dropout=0.1, batch_first=True, **factory_kwargs)

        self.self_attn = MultiheadAttention(self.args.d_model, self.args.nhead, dropout=0.1, batch_first=True, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(self.args.d_model, self.args.dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(self.args.dim_feedforward, self.args.d_model, **factory_kwargs)
        self.linear3 = Linear(self.args.d_model, 2, **factory_kwargs)

        self.norm1 = LayerNorm(self.args.d_model, eps=self.args.layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(self.args.d_model, eps=self.args.layer_norm_eps, **factory_kwargs)
        self.dropout = Dropout(self.args.dropout)
        self.dropout1 = Dropout(self.args.dropout)
        self.dropout2 = Dropout(self.args.dropout)
        self.activation = F.gelu
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(self.args.device))
        # self.loss = nn.CrossEntropyLoss().to(self.args.device)

    def forward(self, query_emb, ques_att_masks, bags_list, bag_pesu_label, batch_loss, retriever, triever_tokenizer, train_flag):
        total_pred = []
        total_logit = []
        select_doc = []
        select_doc_num = []
        for bag, raw_ques_emb, ques_att_mask in zip(bags_list, query_emb, ques_att_masks):
            with torch.no_grad():
                doc_input = triever_tokenizer(bag, truncation=True, return_tensors='pt', max_length=self.args.chunk_size, padding=True).to(self.args.device)
                raw_doc_emb = retriever(**doc_input).last_hidden_state
        
            doc_emb = self.trans_doc(raw_doc_emb, src_key_padding_mask= (1- doc_input["attention_mask"]).bool())

            doc_emb = torch.mean(doc_emb[:, 0, :], dim=0)  # size (bz, len, dim) -> (dim)

            logit = self.linear3(doc_emb) 
            total_logit.append(logit)
            total_pred.append( int(torch.argmax(logit)) )
            select_doc.append( combine_doc([bag[i] for i in range(0, len(bag))]) )
            select_doc_num.append(len(bags_list))

        if train_flag:
            MSL_loss = self.loss(torch.stack(total_logit), torch.LongTensor(bag_pesu_label).to(self.args.device))
        else:
            MSL_loss = 0
            pred_acc = 0

        return MSL_loss, select_doc, select_doc_num, total_pred
    
    def get_att(self, ques_att_mask, doc_att_mask, doc_emb, ques_emb, bz):
        doc_len = doc_emb.size(1) 
        ques_len = ques_emb.size(0) 
        
        # ques_att_mask1 = (1- ques_att_mask.repeat(bz,1)).bool()

        source_mask_expanded = ques_att_mask.unsqueeze(0).unsqueeze(1).expand(bz, doc_len, ques_len) 
        target_mask_expanded = doc_att_mask.unsqueeze(-1).expand(bz, doc_len, ques_len)  
        combined_mask = ((source_mask_expanded * target_mask_expanded)).bool().repeat(self.args.nhead, 1, 1) 
        # combined_mask = F._canonical_mask(mask=combined_mask, mask_name="src_key_padding_mask", other_type=F._none_or_dtype(None), other_name="src_mask", target_type=doc_emb.dtype)

        return combined_mask

    def encoder_layer(self, q, k_v, attn_mask=None, key_padding_mask= None):
        tmp_att, att_weights = self._sa_block(q, k_v, k_v,  attn_mask=attn_mask, key_padding_mask= key_padding_mask)
        q = self.norm1(q + tmp_att)
        q = self.norm2(q + self._ff_block(q))

        att_weights =  torch.where(torch.isnan(att_weights), torch.zeros_like(att_weights), att_weights)
        q =  torch.where(torch.isnan(q), torch.zeros_like(q), q)
        return att_weights, q

    def _sa_block(self, q, k, v, attn_mask=None, key_padding_mask=None) :
        attn_output, attn_output_weights = self.self_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True)
        return self.dropout1(attn_output), attn_output_weights

    def _ff_block(self, x) :
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)