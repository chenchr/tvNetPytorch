from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Forward_Grad(nn.Module):
    def __init__(self, trainable=False):
        super(Forward_Grad, self).__init__()
        self.trainable = trainable
        self.conv_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,2), stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,1), stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.pad_x  = nn.ReplicationPad2d((0,1,0,0))
        self.pad_y  = nn.ReplicationPad2d((0,0,0,1))
        self.conv_x.weight.data = torch.FloatTensor([-1, 1]).view(1,1,1,2)
        self.conv_y.weight.data = torch.FloatTensor([-1, 1]).view(1,1,2,1)
        self.conv_x.weight.requires_grad = trainable
        self.conv_y.weight.requires_grad = trainable
    def forward(self, tensor):
        assert len(tensor.data.shape) == 4

        diff_x = self.conv_x(self.pad_x(tensor))
        diff_y = self.conv_y(self.pad_y(tensor))
        return diff_x, diff_y

class Centered_Grad(nn.Module):
    def __init__(self, trainable=False):
        super(Centered_Grad, self).__init__()
        self.trainable = trainable
        self.conv_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,3), stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.pad_x  = nn.ReplicationPad2d((1,1,0,0))
        self.pad_y  = nn.ReplicationPad2d((0,0,1,1))
        self.conv_x.weight.data = torch.FloatTensor([-0.5, 0, 0.5]).view(1,1,1,3)
        self.conv_y.weight.data = torch.FloatTensor([-0.5, 0, 0.5]).view(1,1,3,1)
        self.conv_x.weight.requires_grad = trainable
        self.conv_y.weight.requires_grad = trainable

    def forward(self, tensor):
        assert len(tensor.data.shape) == 4

        diff_x = self.conv_x(self.pad_x(tensor))
        diff_y = self.conv_y(self.pad_y(tensor))
        return diff_x, diff_y

class Divergence(nn.Module):
    def __init__(self, trainable=False):
        super(Divergence, self).__init__()
        self.trainable = trainable
        self.conv_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,2), stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,1), stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.pad_x  = nn.ReplicationPad2d((1,0,0,0))
        self.pad_y  = nn.ReplicationPad2d((0,0,1,0))
        self.conv_x.weight.data = torch.FloatTensor([-1, 1]).view(1,1,1,2)
        self.conv_y.weight.data = torch.FloatTensor([-1, 1]).view(1,1,2,1)
        self.conv_x.weight.requires_grad = trainable
        self.conv_y.weight.requires_grad = trainable

    def forward(self, tensor_x, tensor_y):
        assert len(tensor_x.data.shape) == 4
        assert len(tensor_y.data.shape) == 4

        diff_x = self.conv_x(self.pad_x(tensor_x))
        diff_y = self.conv_y(self.pad_y(tensor_y))
        div = diff_x + diff_y
        return div

class Gauss_Smooth(nn.Module):
    def __init__(self, trainable=False):
        super(Gauss_Smooth, self).__init__()
        self.trainable = trainable
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,5), stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.pad  = nn.ReplicationPad2d((2,2,2,2))
        ker_init = torch.FloatTensor([[0.000874, 0.006976, 0.01386, 0.006976, 0.000874],
                                            [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
                                            [0.01386, 0.110656, 0.219833, 0.110656, 0.01386],
                                            [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
                                            [0.000874, 0.006976, 0.01386, 0.006976, 0.000874]])
        self.conv.weight.data = ker_init.view(1,1,5,5).type_as(self.conv.weight.data)
        self.conv.weight.requires_grad = trainable

    def forward(self, tensor):
        assert len(tensor.data.shape) == 4

        smoothed = self.conv(self.pad(tensor))
        return smoothed

class Grey_Scale_Image(nn.Module):
    def __init__(self, trainable=False):
        super(Grey_Scale_Image, self).__init__()
        self.trainable = trainable
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1,1), stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv.weight.data = torch.FloatTensor([0.114, 0.587, 0.299]).view(1,3,1,1)
        self.conv.weight.requires_grad = trainable

    def forward(self, tensor):
        assert len(tensor.data.shape) == 4

        out = self.conv(tensor)
        return out

def normalize_images(x1, x2):
    b, c, h, w = x1.data.shape
    min_x1 = x1.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
    max_x1 = x1.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
    min_x2 = x2.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
    max_x2 = x2.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
    
    x1_norm = (x1 - min_x1) / (max_x1 - min_x1) * 255
    x2_norm = (x2 - min_x2) / (max_x2 - min_x2) * 255
    
    return x1_norm, x2_norm

def meshgrid(tensor):
    b, h, w, c = tensor.data.shape
    u = Variable(torch.linspace(-1, 1, w).view(1,1,w,1).type_as(tensor.data).expand(b,h,w,1))
    v = Variable(torch.linspace(-1, 1, h).view(1,h,1,1).type_as(tensor.data).expand(b,h,w,1))
    grid = torch.cat((u, v), dim=3)
    return grid

def transformer(tensor, flow):
    b, h, w, c = flow.data.shape
    flow[:,:,:,0] = flow[:,:,:,0] / (w-1) * 2
    flow[:,:,:,1] = flow[:,:,:,1] / (h-1) * 2
    grid = meshgrid(flow)
    grid += flow
    tensor_out = torch.nn.functional.grid_sample(tensor, grid, padding_mode='border')
    return tensor_out

def warp_image(x, u, v):
    assert len(x.data.shape) == 4
    assert len(u.data.shape) == 4
    assert len(v.data.shape) == 4
    
    delta = torch.cat((u,v), dim=1)
    b, c, h, w = delta.data.shape
    delta = delta.transpose(1,2).transpose(2,3).contiguous()
    return transformer(x, delta)

def zoom_size(height, width, factor):
    new_height = int(float(height) * factor + 0.5)
    new_width = int(float(width) * factor + 0.5)
    return new_height, new_width

def zoom_image(x, new_height, new_width):
    assert len(x.shape) == 4    
    delta = Variable(torch.zeros(x.data.shape[0], new_height, new_width, 2)).type_as(x)
    zoomed_x = transformer(x, delta)
    return zoomed_x

class Per_Iteration(nn.Module):
    def __init__(self, theta, l_t, taut, trainable=False, GRAD_IS_ZERO=1e-12):
        super(Per_Iteration, self).__init__()
        self.theta = theta
        self.l_t = l_t
        self.taut = taut
        self.GRAD_IS_ZERO = GRAD_IS_ZERO
        self.trainable = trainable
        self.div_1 = Divergence(trainable=trainable)
        self.div_2 = Divergence(trainable=trainable)
        self.f_grad1 = Forward_Grad(trainable=trainable)
        self.f_grad2 = Forward_Grad(trainable=trainable)

    def forward(self, rho_c, u1, u2, diff2_x_warp, diff2_y_warp, grad, p11, p12, p21, p22):
        rho = rho_c + diff2_x_warp * u1 + diff2_y_warp * u2 + self.GRAD_IS_ZERO;
            
        d1_1 = Variable(torch.zeros_like(diff2_x_warp.data))
        d2_1 = Variable(torch.zeros_like(diff2_y_warp.data))
        masks1 = rho < (-self.l_t * grad)
        
        d1_1 = masks1.float() * self.l_t * diff2_x_warp
        d2_1 = masks1.float() * self.l_t * diff2_y_warp

        d1_2 = Variable(torch.zeros_like(diff2_x_warp.data))
        d2_2 = Variable(torch.zeros_like(diff2_y_warp.data))
        masks2 = rho > (self.l_t * grad)

        d1_2 = masks2.float() * (-self.l_t * diff2_x_warp)
        d2_2 = masks2.float() * (-self.l_t * diff2_y_warp)
        
        d1_3 = Variable(torch.zeros_like(diff2_x_warp.data))
        d2_3 = Variable(torch.zeros_like(diff2_y_warp.data))
        masks3 = ((rho >= (-self.l_t * grad)).float() * (rho <= (self.l_t * grad)).float() * (grad > self.GRAD_IS_ZERO).float()) > self.GRAD_IS_ZERO
    
        d1_3 = masks3.float() * (-rho / grad * diff2_x_warp)
        d2_3 = masks3.float() * (-rho / grad * diff2_y_warp)
        
        v1 = d1_1 + d1_2 + d1_3 + u1
        v2 = d2_1 + d2_2 + d2_3 + u2
        
        u1 = v1 + self.theta * self.div_1(p11, p12)
        u2 = v2 + self.theta * self.div_2(p21, p22)
        
        u1x, u1y = self.f_grad1(u1)
        u2x, u2y = self.f_grad2(u2)

        p11 = (p11 + self.taut * u1x) / (
            1.0 + self.taut * torch.sqrt(u1x.pow(2) + u1y.pow(2) + self.GRAD_IS_ZERO));
        p12 = (p12 + self.taut * u1y) / (
            1.0 + self.taut * torch.sqrt(u1x.pow(2) + u1y.pow(2) + self.GRAD_IS_ZERO));
        p21 = (p21 + self.taut * u2x) / (
            1.0 + self.taut * torch.sqrt(u2x.pow(2) + u2y.pow(2) + self.GRAD_IS_ZERO));
        p22 = (p22 + self.taut * u2y) / (
            1.0 + self.taut * torch.sqrt(u2x.pow(2) + u2y.pow(2) + self.GRAD_IS_ZERO));
        
        return u1, u2, p11, p12, p21, p22, rho

class Per_Warp(nn.Module):
    def __init__(self, theta, l_t, taut, iteration_num, trainable=False, GRAD_IS_ZERO=1e-12):
        super(Per_Warp, self).__init__()
        self.theta = theta
        self.l_t = l_t
        self.taut = taut
        self.GRAD_IS_ZERO = GRAD_IS_ZERO
        self.trainable = trainable
        self.iter_num = iteration_num
        self.iteration_list = nn.ModuleList([Per_Iteration(theta=theta, 
                                                           l_t=l_t, 
                                                           taut=taut, 
                                                           trainable=trainable, 
                                                           GRAD_IS_ZERO=GRAD_IS_ZERO
                                                           ) 
                                                           for i in range(iteration_num)])

    def forward(self, x1, x2, u1, u2, diff2_x, diff2_y, p11, p12, p21, p22):
        x2_warp = warp_image(x2, u1, u2)
        
        diff2_x_warp = warp_image(diff2_x, u1, u2)
        diff2_y_warp = warp_image(diff2_y, u1, u2)

        diff2_x_sq = diff2_x_warp.pow(2)
        diff2_y_sq = diff2_y_warp.pow(2)

        grad = diff2_x_sq + diff2_y_sq + self.GRAD_IS_ZERO
        
        rho_c = x2_warp - diff2_x_warp * u1 - diff2_y_warp * u2 - x1

        for per_iter in self.iteration_list:
            u1, u2, p11, p12, p21, p22, rho = per_iter(rho_c, u1, u2, diff2_x_warp, diff2_y_warp, grad, p11, p12, p21, p22)
        
        return u1, u2, p11, p12, p21, p22, rho

class Per_Scale(nn.Module):
    def __init__(self, 
                 tau=0.25,  # time step
                 lbda=0.15,  # weight parameter for the data term
                 theta=0.3,  # weight parameter for (u - v)^2
                 warps=5,  # number of warpings per scale
                 max_iterations=5,  # maximum number of iterations for optimization)
                 trainable = False,
                 GRAD_IS_ZERO=1e-12
                ):
        super(Per_Scale, self).__init__()
        self.tau = tau
        self.lbda = lbda
        self.theta = theta
        self.warp_num = warps
        self.iter_num = max_iterations
        self.tarinable = trainable
        self.GRAD_IS_ZERO = GRAD_IS_ZERO
        self.l_t = lbda * theta
        self.taut = tau / theta
        self.center_grad = Centered_Grad(trainable=trainable)
        self.warp_list = nn.ModuleList([Per_Warp(theta=theta, 
                                                 l_t=self.l_t, 
                                                 taut=self.taut, 
                                                 iteration_num=max_iterations, 
                                                 trainable=trainable, 
                                                 GRAD_IS_ZERO=GRAD_IS_ZERO
                                                 )
                                                 for i in range(self.warp_num)])

    def forward(self, x1, x2, u1, u2):
        diff2_x, diff2_y = self.center_grad(x2)

        p11, p12, p21, p22 = [Variable(torch.zeros_like(x1.data)) for i in range(4)]

        for per_warp in self.warp_list:
            u1, u2, p11, p12, p21, p22, rho = per_warp(x1, x2, u1, u2, diff2_x, diff2_y, p11, p12, p21, p22)

        return u1, u2, rho

class TvNet(nn.Module):
    def __init__(self,
                 min_size,
                 tau=0.25,  # time step
                 lbda=0.15,  # weight parameter for the data term
                 theta=0.3,  # weight parameter for (u - v)^2
                 warps=5,  # number of warpings per scale
                 zfactor=0.5,  # factor for building the image piramid
                 max_scales=5,  # maximum number of scales for image piramid
                 max_iterations=5,  # maximum number of iterations for optimization):
                 trainable=False, 
                 GRAD_IS_ZERO=1e-12
                 ):
        super(TvNet, self).__init__()
        self.tau = tau
        self.lbda = lbda
        self.theta = theta
        self.warp_num = warps
        self.zfactor = zfactor
        self.max_scales = max_scales
        self.iter_num = max_iterations
        self.trainable = trainable
        self.GRAD_IS_ZERO = GRAD_IS_ZERO

        height, width = min_size
        zfactor = float(zfactor)
        n_scales = 1 + np.log(np.sqrt(height ** 2 + width ** 2) / 4.0) / np.log(1 / zfactor);
        self.n_scales = min(n_scales, max_scales)
        self.scale_list = nn.ModuleList([Per_Scale(tau=tau,  # time step
                                         lbda=lbda,  # weight parameter for the data term
                                         theta=theta,  # weight parameter for (u - v)^2
                                         warps=warps,  # number of warpings per scale
                                         max_iterations=max_iterations,  # maximum number of iterations for optimization)
                                         trainable=trainable,
                                         GRAD_IS_ZERO=GRAD_IS_ZERO
                                        ) 
                                    for i in range(self.n_scales)])
        self.to_grey1 = Grey_Scale_Image(trainable=trainable)
        self.to_grey2 = Grey_Scale_Image(trainable=trainable)
        self.smooth1 = Gauss_Smooth(trainable=trainable)
        self.smooth2 = Gauss_Smooth(trainable=trainable)

    def forward(self, x1, x2):
        grey_x1 = self.to_grey1(x1)
        grey_x2 = self.to_grey2(x2)
        norm_imgs = normalize_images(grey_x1, grey_x2)

        smooth_x1 = self.smooth1(norm_imgs[0])
        smooth_x2 = self.smooth2(norm_imgs[1])

        height, width = x1.data.shape[2:]
        ss = self.n_scales - 1

        
        for per_scale in self.scale_list:
            down_sample_factor = self.zfactor ** ss
            down_height, down_width = zoom_size(height, width, down_sample_factor)

            down_x1 = zoom_image(smooth_x1, down_height, down_width)
            down_x2 = zoom_image(smooth_x2, down_height, down_width)
            
            if ss == self.n_scales - 1:
                u1 = Variable(torch.FloatTensor(x1.data.shape[0], 1, down_height, down_width).type_as(x1.data).zero_(), requires_grad=self.trainable)
                u2 = Variable(torch.FloatTensor(x1.data.shape[0], 1, down_height, down_width).type_as(x1.data).zero_(), requires_grad=self.trainable)
        
            u1, u2, rho = per_scale(down_x1, down_x2, u1, u2)

            if ss == 0:
                break
            
            up_sample_factor = self.zfactor ** (ss - 1)
            up_height, up_width = zoom_size(height, width, up_sample_factor)
            u1 = zoom_image(u1, up_height, up_width) / self.zfactor
            u2 = zoom_image(u2, up_height, up_width) / self.zfactor
            
            ss = ss - 1

        return u1, u2, rho