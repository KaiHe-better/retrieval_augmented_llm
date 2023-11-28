
import torch.nn as nn


class Dragon(nn.Module):
    def __init__(self, query_encoder,  context_encoder):
        nn.Module.__init__(self)
        self.query_encoder = query_encoder
        self.context_encoder = context_encoder
 
 

