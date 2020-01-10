from __future__ import absolute_import
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import warnings
from .LOSSES import LOSSES

class DRO_TOPK(nn.Module):
    # Truncated DRO for p_choice = 1

    ####TODO: finish two methods, the third methods need to be done.
    def __init__(self, alpha=40, margin=0.5, beta=0,  K = 5,
                 select_TOPK_all = 1, loss = "margin_loss", **kwargs):

        # self.choose = 1, DRO over batch
        # self.choose = 2, DRO over class
        # self.choose = 3, DRO over each anchor.
        super(DRO_TOPK, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.select_TOPK_all = select_TOPK_all
        self.K = K
        self.loss = loss
        self.LOSSES = LOSSES(alpha = self.alpha, beta = self.beta, margin = self.margin)
        print("loss:", self.loss, 'alpha:', self.alpha, 'beta:', self.beta,
              "margin:", self.margin, 'K', self.K)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets.cuda()


        eyes_ = Variable(torch.eye(n, n)).cuda()

        #print("eyes_.dtype, targets.dtype:", eyes_.dtype, targets.dtype)

        pos_mask = targets.expand(n,n).eq(targets.expand(n,n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask # negative pairs index mat
        pos_mask = pos_mask - eyes_.eq(1) # positive pairs index mat

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        #method_to_call = ,
        #result = method_to_call()

        myloss = getattr(self.LOSSES, self.loss)
        #print("function call", myloss)
        pos_loss, neg_loss = myloss(pos_sim, neg_sim)

        all_loss = torch.cat((pos_loss, neg_loss), 0)
        num_of_zeros = torch.sum(torch.eq(all_loss, 0))


        if self.select_TOPK_all == 1:

            if self.K * 4 > all_loss.size()[0]:
                loss = torch.mean(all_loss)
                warnings.warn("K larger than the total number of pairs in the batch. The loss is calculated using all the pairs by default.")
            else:
                top_loss, _ = torch.topk(all_loss, self.K * 4, largest=True)
                loss = torch.mean(top_loss)

        else:
            if self.K * 2 <= pos_loss.size()[0] and self.K * 2 <= neg_loss.size()[0]:
                top_pos_loss, _ = torch.topk(pos_loss, self.K * 2, largest=True)
                top_neg_loss, _ = torch.topk(neg_loss, self.K * 2, largest=True)
                loss = (torch.sum(top_pos_loss)+torch.sum(top_neg_loss))/(self.K*4)
            else:
                top_neg_loss, _ = torch.topk(neg_loss, pos_loss.size()[0], largest=True)
                loss = (torch.sum(pos_loss) + torch.sum(top_neg_loss)) / (2*pos_loss.size()[0])
                warnings.warn("K either larger than the number of positive pairs or larger than negative pairs. The loss is calculated using all the positive pairs and the same number amount of negative pairs which have the largest losses by default.")

        mean_neg_sim = torch.mean(pos_sim).item()
        mean_pos_sim = torch.mean(neg_sim).item()

        return loss, num_of_zeros.item(), mean_neg_sim, mean_pos_sim


    # def calibration_check_p(self, p):
    #     # remove nan, all o p, and make sure sum(p) = 1
    #
    #     if (torch.sum(torch.isnan(p)) != 0): # remove nan
    #         p[torch.isnan(p)] == 1 / torch.sum(torch.isnan(p))
    #     if (torch.sum(p) != 0):  # make sure the sum of p equal to 1.
    #         p = p / (torch.sum(p))
    #     return p
    #
    #
    # def truncate_p(self, np_p, K_thresh):
    #     order_np_p = -np.sort(-np_p)  # ordering from large to small
    #     np_p[np_p < order_np_p[K_thresh]] = 0
    #     np_p = np_p / sum(np_p)
    #     return np_p
def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    DRO_TOPK(K = 1000, select_TOPK_all=2)(inputs, targets)


if __name__ == '__main__':
    main()
    print(LOSSES.__dict__.items())
    print('Congratulations to you!')


