import sys
import torch.nn as nn
from utils1.metric import dc_mean

# 指标累计器
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        # self.reset()
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# log打印
class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, label):
        return 1 - dc_mean(pred, label)


# 损失计算，默认权重为1
class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, label):
        bceloss = self.bce(pred, label)
        diceloss = self.dice(pred, label)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss