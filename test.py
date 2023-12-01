import torch.nn.functional as F
import torch.nn as nn
import torch


kl_loss = nn.KLDivLoss(reduction="batchmean")
# input should be a distribution in the log space
input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)


target = torch.rand(3, 5)
# Sample a batch of distributions. Usually this would come from the dataset
target = F.softmax(target, dim=1)
output = kl_loss(input, target)
print(output)

kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
log_target = F.log_softmax(target, dim=1)
output = kl_loss(input, log_target)
print(output)