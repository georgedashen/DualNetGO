import torch
import torch.nn as nn
import torch.nn.functional as F

class pretrainLossOptimized_MDA(nn.Module):
    def __init__(self, clip=0.05, eps=1e-5):
        super(pretrainLossOptimized_MDA, self).__init__()
        self.clip = clip
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None
        self.LARGE_NUM = 1e9
    def forward(self, ori, rec, hs):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        raw_h1: logits h1
        raw_h2: logits h2
        """
        
        x1 = ori[0]
        x2 = ori[1]
        x3 = ori[2]
        x4 = ori[3]
        x5 = ori[4]
        x6 = ori[5]
        x7 = ori[6]
        
        rec_x1 = rec[0].squeeze()
        rec_x2 = rec[1].squeeze()
        rec_x3 = rec[2].squeeze()
        rec_x4 = rec[3].squeeze()
        rec_x5 = rec[4].squeeze()
        rec_x6 = rec[5].squeeze()
        rec_x7 = rec[6].squeeze()
        
        recon_loss_1 = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(rec_x1, x1, reduction='none'), dim=1))
        recon_loss_2 = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(rec_x2, x2, reduction='none'), dim=1))
        recon_loss_3 = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(rec_x3, x3, reduction='none'), dim=1))
        recon_loss_4 = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(rec_x4, x4, reduction='none'), dim=1))
        recon_loss_5 = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(rec_x5, x5, reduction='none'), dim=1))
        recon_loss_6 = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(rec_x6, x6, reduction='none'), dim=1))
        recon_loss_7 = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(rec_x7, x7, reduction='none'), dim=1))
        recon_loss = recon_loss_1 + recon_loss_2 + recon_loss_3 + recon_loss_4 + recon_loss_5 + recon_loss_6 + recon_loss_7
        recon_loss += 0 * torch.sum(hs)
        return recon_loss

