from . import DRO
from . import models




def DRO_AVG(*args, **kwargs):
    ave_loss = losses.create("DRO_AVG", margin=args.margin, alpha=args.alpha, base=args.loss_base).cuda()

def DRO_SAM(*args, **kwargs):

    return

def DRO_Grouping(*args, **kwargs):

    return


def Top_K(*args, **kwargs):
    return

