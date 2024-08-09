import torch
import torch.nn as nn
import torchvision


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args

        model = torchvision.models.resnet34(pretrained = True)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

            
    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048