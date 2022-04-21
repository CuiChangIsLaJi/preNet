import os
import torch
from torch import nn
from torch.nn import BCELoss
from torch.nn import functional as F
import cv2
import numpy as np
from matplotlib import pyplot as plt

class GradCAM:
    def __init__(self,model,target_layers,info):
        self.model = model
        self.model.eval()

        if isinstance(target_layers,list):
            for layer in self.target_layers:
                layer.register_forward_hook(self.__forward_hook)
                layer.register_full_backward_hook(self.__backward_hook)
        elif isinstance(target_layers,nn.Module):
            target_layers.register_forward_hook(self.__forward_hook)
            target_layers.register_full_backward_hook(self.__backward_hook)
        else:
            raise TypeError("Argument 'target_layers' should be either a torch.nn.Module object or a list.")

        if info not in ["seq","struct"]:
            raise ValueError("Argument 'info' should be either 'seq' or 'struct'")
        self.info = info

        self.feature_size = None
        self.grads = []
        self.fmaps = []
        self.cams = []
    def __forward_hook(self,module,in_feature,out_feature):
        self.fmaps.append(out_feature.squeeze())
    def __backward_hook(self,module,in_grad,out_grad):
        self.grads.append(out_grad[0].detach().squeeze())
    def __compute_cam(self,grad,fmap):
        """
        grad: Tensor in shape (C,H,W)
        fmap: Tensor in shape (C,H,W)
        return: Tensor in shape (H,W)
        """
        height,weight = tuple(grad.size()[1:])
        cam = torch.zeros(height,weight)
        alpha = F.avg_pool2d(grad,(height,weight)).squeeze()
        for k,alpha_k in enumerate(alpha):
            cam += alpha_k * fmap[k]
        return F.relu(cam)
    def __show_cam(self,cam_show,X,path,name):
        plt.axis("off")
        plt.imshow(cam_show)
        plt.savefig(os.path.join(path,name))
        plt.cla()
    def forward(self,X_seq,X_struct,label,path,name):
        self.feature_size = (X_seq.size()[-1],X_seq.size()[-2])
        score = self.model(X_seq,X_struct).flatten()

        self.model.zero_grad()
        loss = BCELoss()(score,label)
        loss.backward()

        for grad,fmap in zip(self.grads,self.fmaps):
            cam = self.__compute_cam(grad,fmap).detach().numpy()
            self.cams.append(cam)
            cam_show = cv2.resize(cam,self.feature_size)
            X = X_seq if self.info=="seq" else X_struct
            X = X.squeeze()
            self.__show_cam(cam_show,X,path,name)

        self.fmaps.clear()
        self.grads.clear()

        return score.item() if self.info=="seq" else None
