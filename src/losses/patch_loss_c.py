import torch
import torch.nn as nn
from torch.nn import functional as F
    
class PatchesKernel3D(nn.Module):
    def __init__(self, kernelsize, kernelstride, kernelpadding=0):
        super(PatchesKernel3D, self).__init__()
        kernel = torch.eye(kernelsize ** 2).\
            view(kernelsize ** 2, 1, kernelsize, kernelsize)
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel,
                                   requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(kernelsize ** 2),
                                 requires_grad=False)
        self.kernelsize = kernelsize
        self.stride = kernelstride
        self.padding = kernelpadding

    def forward(self, x):
        batchsize = x.shape[0]
        channels = x.shape[1]
        x = x.reshape(batchsize*channels, x.shape[-2], x.shape[-1]).unsqueeze(1)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, channels, -1, self.kernelsize ** 2).permute(0, 2, 1, 3)
        return x

class patchLoss3DXD(nn.Module):
    """Define patch loss

    Args:
        kernel_sizes (list): add (x, y) in the list.
        loss_weight (float): Loss weight. Default: 1.0.
    """
    def __init__(self, kernel_sizes=[2, 4], use_std_to_force=True):
        super(patchLoss3DXD, self).__init__()
        self.kernels = kernel_sizes
        self.use_std_to_force = use_std_to_force

    def forward(self, preds, labels):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            labels (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        loss = 0.
        for _kernel in self.kernels:
            _patchkernel = PatchesKernel3D(_kernel, _kernel//2 + 1).to('cuda') # create instance
            preds_trans = _patchkernel(preds)                                  # [N, patch_num, channels, patch_len ** 2]
            labels_trans = _patchkernel(labels)                                # [N, patch_num, channels, patch_len ** 2]
            preds_trans = preds_trans.reshape(-1, preds_trans.shape[-1])       # [N * patch_num * channels, patch_len ** 2]
            labels_trans = labels_trans.reshape(-1, labels_trans.shape[-1])    # [N * patch_num * channels, patch_len ** 2]
            x = torch.clamp(preds_trans, 0.000001, 0.999999)
            y = torch.clamp(labels_trans, 0.000001, 0.999999)
            dot_x_y = torch.einsum('ik,ik->i',x,y)      
            if self.use_std_to_force == False:
                cosine0_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1)))
                loss = loss + torch.mean((1-cosine0_x_y)) # y = 1-x
            else:
                dy = torch.std(labels_trans*10, dim=1)
                cosine_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1)))
                cosine_x_y_d = torch.mul((1-cosine_x_y), dy) # y = (1-x) dy
                loss = loss + torch.mean(cosine_x_y_d) 
        return loss