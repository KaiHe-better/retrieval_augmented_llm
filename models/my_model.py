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
        # self.loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(self.args.device))
        self.loss = nn.CrossEntropyLoss().to(self.args.device)

    def forward(self, query_emb, ques_att_masks, bags_list, bag_pesu_label, batch_loss, retriever, triever_tokenizer, train_flag):
        total_pred = []
        total_logit = []
        select_doc = []
        select_doc_num = []
        for bag, raw_ques_emb, ques_att_mask in zip(bags_list, query_emb, ques_att_masks):
            print(bag[0][:10])
            with torch.no_grad():
                doc_input = triever_tokenizer(bag, truncation=True, return_tensors='pt', max_length=self.args.chunk_size, padding=True).to(self.args.device)
                raw_doc_emb = retriever(**doc_input).last_hidden_state
            
            bz = raw_doc_emb.size(0)

            # doc_att_mask = doc_input['attention_mask']
            # attn_mask = self.get_att(ques_att_mask, doc_att_mask, raw_doc_emb, raw_ques_emb, bz)

            raw_ques_emb = raw_ques_emb.repeat(bz,1,1)
            ques_att_mask = (1- ques_att_mask.repeat(bz,1)).bool()

            _, doc_emb = self.encoder_layer(raw_doc_emb, raw_ques_emb, attn_mask=None, key_padding_mask= ques_att_mask )
            # _, doc_emb = self.encoder_layer(raw_doc_emb, raw_ques_emb, attn_mask=attn_mask, key_padding_mask= None )
            doc_emb = torch.mean(doc_emb[:, 0, :], dim=0)  # size (bz, len, dim) -> (dim)

            logit = self.linear3(doc_emb) 
            total_logit.append(logit)
            total_pred.append( int(torch.argmax(logit)) )
            select_doc.append( combine_doc([bag[i] for i in range(0, len(bag))]) )
            select_doc_num.append(len(bag))

        if train_flag:
            MSL_loss = self.loss(torch.stack(total_logit), torch.LongTensor(bag_pesu_label).to(self.args.device) )
        else:
            MSL_loss = 0
            pred_acc = 0

        return MSL_loss, select_doc, select_doc_num, total_pred
    
    def get_att(self, ques_att_mask, doc_att_mask, doc_emb, ques_emb, bz):
        
        ques_att_mask = ques_att_mask.unsqueeze(0).repeat(bz, 1)
        attn_mask = torch.einsum('bl,bs->bls', doc_att_mask, ques_att_mask)  # (batch_size, L, S)
        attn_mask = 1- attn_mask.repeat_interleave(self.args.nhead, dim=0)  # (batch_size * num_heads, L, S)

        return attn_mask.bool()

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