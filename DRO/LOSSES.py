
import torch


class LOSSES:

    def __init__(self, margin = 0.1, alpha = 40, beta = 0, **kwargs):

        self.margin = margin
        self.alpha = alpha
        self.beta = beta

    def margin_loss(self, pos_pair, neg_pair):
        '''
        Equation: l = (alpha + y_{ij}(self.margin-s_{ij}^+))_+
        '''
        pos_loss = torch.max(self.alpha + self.margin - pos_pair, torch.zeros_like(pos_pair))
        neg_loss = torch.max(self.alpha + neg_pair - self.margin, torch.zeros_like(neg_pair))
        return pos_loss, neg_loss


    def lifted_structure(self, pos_pair, neg_pair):
        pos_loss = 2.0 / self.beta * torch.log(torch.sum(torch.exp(-self.beta * pos_pair)))
        neg_loss = 2.0 / self.alpha * torch.log(torch.sum(torch.exp(self.alpha * neg_pair)))
        return pos_loss, neg_loss


    def binomial(self, pos_pair, neg_pair):
        pos_loss = 2.0 / self.beta * torch.log(1 + torch.exp(-self.beta * (pos_pair - self.margin)))
        neg_loss = 2.0 / self.alpha * torch.log(1 + torch.exp(self.alpha * (neg_pair - self.margin)))
        return pos_loss, neg_loss

    def constrastive(self, pos_pair, neg_pair):
        neg_loss = torch.masked_select(neg_pair, neg_pair > self.margin)
        pos_loss = -pos_pair + 1

        return pos_loss, neg_loss







