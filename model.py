import torch
import torch.nn as nn
from image import ImageEncoder
import torch.nn.functional as F
import numpy as np

def getVac(alpha):
    s = torch.sum(alpha, dim=1, keepdim=True)
    vacuity = alpha.shape[1] / s

    return vacuity

# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ll_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean(A + B)


def cons_gradient(p, alpha1, alpha2, c):
    label = F.one_hot(p, num_classes=c)
    S1 = torch.sum(alpha1, dim=1, keepdim=True)
    S2 = torch.sum(alpha2, dim=1, keepdim=True)
    u1 = c/S1
    u2 = c/S2
    g1 = label*(1/alpha1-u1)
    g2 = label*(1/alpha2-u2)
    
    l2 =torch.abs(g1-g2).mean()
    
    return l2

    
class UMC(nn.Module):
    def __init__(self, args):
        super(UMC, self).__init__()
        self.args = args
        self.view1enc = ImageEncoder(args)
        self.view2enc = ImageEncoder(args)
        self.view3enc = ImageEncoder(args)
        view_last_size = args.img_hidden_sz
        
        self.clf_view1 = nn.ModuleList()
        self.clf_view2 = nn.ModuleList()
        self.clf_view3 = nn.ModuleList()
        
        self.clf_view1 = nn.Linear(view_last_size, args.n_classes)
        self.clf_view2 = nn.Linear(view_last_size, args.n_classes)
        self.clf_view3 = nn.Linear(view_last_size, args.n_classes)


    def ABF_Combin(self, view_a):
        e_c = 0
        c = 0
        view_e = dict()

        for v_num in range(len(view_a)):
            view_e[v_num] = view_a[v_num] - 1
            u = getVac(view_a[v_num])
            c += (1 - u)
            e_c += (1 - u) * view_e[v_num]

        e_final = e_c / c
        alpha_a = e_final + 1

        return alpha_a

    def forward(self, view1, view2, view3):
        view1 = self.view1enc(view1)
        view1 = torch.flatten(view1, start_dim=1)
        view2 = self.view1enc(view2)
        view2 = torch.flatten(view2, start_dim=1)
        view3 = self.view1enc(view3)
        view3 = torch.flatten(view3, start_dim=1)
        
        view1_out = self.clf_view1(view1)
        view2_out = self.clf_view2(view2)
        view3_out = self.clf_view3(view3)


        view1_evidence, view2_evidence, view3_evidence = torch.exp(torch.clamp(view1_out,-10,10)), torch.exp(torch.clamp(view2_out,-10,10)), torch.exp(torch.clamp(view3_out,-10,10))
        view1_alpha, view2_alpha, view3_alpha = view1_evidence+1, view2_evidence+1, view3_evidence+1
        view_a = [view1_alpha, view2_alpha, view3_alpha]
        fusion_alpha = self.ABF_Combin(view_a)
        return view1_alpha, view2_alpha, view3_alpha, fusion_alpha
    