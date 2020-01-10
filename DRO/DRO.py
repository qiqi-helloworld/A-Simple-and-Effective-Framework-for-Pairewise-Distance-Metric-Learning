
from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from .LOSSES import LOSSES


class DRO(nn.Module):
    # deterministiclly set p to each pair
    # Solving inf/inf problem for p_choice=3
    # Exisiting Problem: Out of Cuda Memory for Choice_1
    def __init__(self, alpha=0.2, beta=0, margin=0.5, loss = "margin_loss",
                 p_lambda= 0.1, p_lambda_neg=0.1,  K = 5, p_choice = 1,
                 plambda_eq = 1,
                 **kwargs):
        # self.choose = 1, DRO over batch
        # self.choose = 2, DRO by class: You are able to set different p_lambda
        # self.choose = 3, DRO by each anchor.
        super(DRO, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.p_lambda = p_lambda
        self.p_lambda_neg = p_lambda_neg
        self.p_choice = p_choice
        self.K = K
        self.loss = loss
        if plambda_eq:
            self.p_lambda_neg  = self.p_lambda
        self.LOSSES = LOSSES(alpha = self.alpha, beta = self.beta, margin = self.margin)
        print("DRO, p_lambda:", self.p_lambda, "p_lambda_neg:", self.p_lambda_neg, "loss", self.loss, 'p_choice', self.p_choice)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        targets = targets.cuda()
        eyes_ = Variable(torch.eye(n,n)).cuda()

        pos_mask = targets.expand(n,n).eq(targets.expand(n,n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        myloss = getattr(self.LOSSES, self.loss)

        #pos_loss = torch.max(0.2 + 0.5 - pos_sim, torch.zeros_like(pos_sim))
        #neg_loss = torch.max(0.2 + neg_sim - 0.5, torch.zeros_like(neg_sim))

        pos_loss, neg_loss = myloss(pos_sim, neg_sim)

        all_loss = torch.cat((pos_loss, neg_loss), 0)
        num_of_zeros = torch.sum(torch.eq(all_loss, 0))
        #print("num_of_zeros:", num_of_zeros)

        if self.p_choice == 1:
            # pick among all pairs.
            all_loss = torch.cat((pos_loss, neg_loss), 0)
            lambda_all_loss = all_loss/self.p_lambda

            #calibration
            if torch.sum(torch.exp(lambda_all_loss) - 1) != 0:
                p = (torch.exp(lambda_all_loss) - 1)/torch.sum(torch.exp(lambda_all_loss) - 1)
            else:
                p = (torch.exp(lambda_all_loss))/(torch.sum(torch.exp(lambda_all_loss)))
            np_p = p.data.cpu().numpy()

            order_np_p = -np.sort(-np_p)

            np_p[np_p < order_np_p[self.K*4]] = 0
            np_p = np_p/sum(np_p)

            index = np.random.choice(all_loss.size()[0], self.K*4, replace=True, p = np_p)
            selected_loss = all_loss[index]
            loss = torch.mean(selected_loss)
        elif self.p_choice == 2:
            # pick for positive pairs or negative pairs, separately.
            lambda_pos_loss = pos_loss/self.p_lambda
            lambda_neg_loss = neg_loss/self.p_lambda_neg

            if torch.sum(torch.exp(lambda_pos_loss - 1))!=0:
                p_pos = (torch.exp(lambda_pos_loss) - 1)/torch.sum(torch.exp(lambda_pos_loss) - 1)
            else:
                p_pos = (torch.exp(lambda_pos_loss))/torch.sum(torch.exp(lambda_pos_loss))

            if torch.sum(torch.exp(lambda_neg_loss - 1)) != 0:
                p_neg = (torch.exp(lambda_neg_loss) - 1) / torch.sum(torch.exp(lambda_neg_loss) - 1)
            else:
                p_neg = (torch.exp(lambda_neg_loss)) / torch.sum(torch.exp(lambda_neg_loss))


            np_p_pos = p_pos.data.cpu().numpy()
            np_p_neg = p_neg.data.cpu().numpy()

            p_index = np.random.choice(lambda_pos_loss.size()[0], self.K*2, replace=True, p=np_p_pos)
            n_index = np.random.choice(lambda_neg_loss.size()[0], self.K*2, replace=True, p=np_p_neg)

            selected_pos_loss = pos_loss[p_index]
            selected_neg_loss =neg_loss[n_index]

            loss = (torch.mean(selected_pos_loss) + torch.mean(selected_neg_loss))/2

        else:
            #pick for each anchor
            loss = list()
            for i in range(n):
                pos_pair = torch.masked_select(sim_mat[i], targets == targets[i])
                pos_pair = torch.masked_select(pos_pair, pos_pair < 1)
                neg_pair = torch.masked_select(sim_mat[i], targets != targets[i])

                pos_pair_loss = torch.max(0.2 + self.margin - pos_pair, torch.zeros_like(pos_pair))
                neg_pair_loss = torch.max(0.2 + neg_pair - self.margin, torch.zeros_like(neg_pair))

                lambda_pos_pair_loss = pos_pair_loss/self.p_lambda
                lambda_neg_pair_loss = neg_pair_loss/self.p_lambda_neg

                #calibration
                if torch.sum(torch.exp(lambda_pos_pair_loss) - 1) != 0:
                    p_pos = (torch.exp(lambda_pos_pair_loss) - 1)/torch.sum(torch.exp(lambda_pos_pair_loss) - 1)
                else:
                    p_pos = (torch.exp(lambda_pos_pair_loss))/(torch.sum(torch.exp(lambda_pos_pair_loss)))

                if torch.sum(torch.exp(lambda_neg_pair_loss) - 1) != 0:
                    p_neg = (torch.exp(lambda_neg_pair_loss) - 1) / torch.sum(torch.exp(lambda_neg_pair_loss) - 1)
                else:
                    p_neg = (torch.exp(lambda_neg_pair_loss)) / (torch.sum(torch.exp(lambda_neg_pair_loss)))

                np_p_pos = p_pos.data.cpu().numpy()
                np_p_neg = p_neg.data.cpu().numpy()
                if self.K > pos_pair_loss.size()[0]:
                    p_index = np.random.choice(pos_pair_loss.size()[0], pos_pair_loss.size()[0], replace=True, p= np_p_pos)
                    n_index = np.random.choice(neg_pair_loss.size()[0], neg_pair_loss.size()[0], replace=True, p= np_p_neg)
                else:
                    p_index = np.random.choice(pos_pair_loss.size()[0], self.K, replace=True, p=np_p_pos)
                    n_index = np.random.choice(neg_pair_loss.size()[0], self.K, replace=True, p= np_p_neg)

                selected_pos_loss = pos_pair_loss[p_index]
                selected_neg_loss = neg_pair_loss[n_index]

                loss.append(torch.sum(selected_pos_loss) + torch.sum(selected_neg_loss))

            if (self.K > pos_pair_loss.size()[0]):
                loss = sum(loss)/(n*pos_pair_loss.size()[0]*2)
            else:
                loss = sum(loss)/(n*self.K*2)

        mean_neg_sim = torch.mean(pos_sim).item()
        mean_pos_sim = torch.mean(neg_sim).item()

        return loss, num_of_zeros.item(), mean_neg_sim, mean_pos_sim

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
    #print("W:", w, inputs)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(DRO()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


