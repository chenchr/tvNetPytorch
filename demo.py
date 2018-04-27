import torch
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt
import math
from tvnet import TvNet
import time
from scipy.misc import imsave

def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D

def flow2rgb(u, v, max_value):
    flow_map = torch.cat((u,v), dim=1)
    flow_map = flow_map.data[0].cpu().numpy()
    _, h, w = flow_map.shape
    flow_map[:,(flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    rgb_map = np.ones((h,w,3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:,:,0] += normalized_flow_map[0]
    rgb_map[:,:,1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:,:,2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

path_im1 = '/home/chenchr/code/FlowNet/models/flownet/data/0000005-img0.png'
path_im2 = '/home/chenchr/code/FlowNet/models/flownet/data/0000005-img1.png'
path_im1 = 'img1.png'
path_im2 = 'img2.png'
path_flow = '/home/chenchr/code/FlowNet/models/flownet/data/0000005-gt.flo'
im1_np = imread(path_im1).astype(np.float32)
im2_np = imread(path_im2).astype(np.float32)
im1, im2 = [torch.from_numpy(np.transpose(im, (2,0,1)).astype(np.float32)).unsqueeze(0) for im in [im1_np, im2_np]]
flow = torch.from_numpy(load_flo(path_flow)).unsqueeze(0)
im1, im2, flow = [torch.autograd.Variable(t, volatile=True) for t in [im1, im2, flow]]

the_net = TvNet(
                 min_size=(500,500),
                 tau=0.25,  # time step
                 lbda=0.15,  # weight parameter for the data term
                 theta=0.3,  # weight parameter for (u - v)^2
                 warps=5,  # number of warpings per scale
                 zfactor=0.5,  # factor for building the image piramid
                 max_scales=5,  # maximum number of scales for image piramid
                 max_iterations=50,  # maximum number of iterations for optimization):
                 trainable=False, 
                 GRAD_IS_ZERO=1e-12
                 )
use_cuda = False
if use_cuda:
    im1, im2, the_net = [t.cuda() for t in [im1, im2, the_net]]

time_begin = time.time()
# for name, param in the_net.named_modules():
#     print(name)
for i in range(1):
    u1, u2, rho = the_net(im1, im2)
time_end = time.time()
u1_np = u1.data[0][0].cpu().numpy()
u2_np = u2.data[0][0].cpu().numpy()

time_cost = time_end - time_begin
print('time cost: {}'.format(time_cost))
flow_np = flow2rgb(u1, u2, 1)
plt.imshow(flow_np)
plt.show()
imsave('flow.png', flow_np)