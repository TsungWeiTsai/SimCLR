import torch
from torch import nn

class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch  
    '''
    def __init__(self, negM, T, batchSize):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        self.diag_mat = 1 - torch.eye(batchSize*2).cuda()
        
    def forward(self, x):
        batchSize = x.size(0)
        
        reordered_x = torch.cat( (x.narrow(0,batchSize//2,batchSize//2),\
                x.narrow(0,0,batchSize//2)), 0)
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_()

        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*self.diag_mat
        if self.negM==1:
            all_div = all_prob.sum(1)
        else:
            all_div = (all_prob.sum(1) - pos)*self.negM + pos

        lnPmt = torch.div(pos, all_div)

        Pon_div = all_div.repeat(batchSize,1)  # 256x1 --> 256x256
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)
        
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum)/batchSize
        return loss, torch.exp(lnPmt)
