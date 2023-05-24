#參考 https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/models/non_local.py
import torch
from torch import nn
from torch.nn import functional as F
import datetime
import psutil



class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )

    def forward(self,fast,slow):
        def tensor_reshape(self,in_tensor,reshape_hw):
            start=datetime.datetime.now()
            #set target h and w
            target_height, target_width = reshape_hw, reshape_hw
            #set output size
            out_tensor = torch.zeros(1, 80, 4, target_height, target_width)
            #for slice process bilinear
            for d in range(in_tensor.size(2)):
                slice_tensor = in_tensor[:, :, d, :, :]
                interpolated_slice = F.interpolate(slice_tensor, size=(target_height, target_width), mode='bilinear')
                out_tensor[:, :, d, :, :] = interpolated_slice
            print(out_tensor.shape)
            end=datetime.datetime.now()
            duration = end - start
            print("total sec", duration.total_seconds())
            return out_tensor.cuda()
            
        def cpu_gpu_memory(self,log):
            print('{}\n{}\nCPU:{:.2f}\nGPU:{:.2f}'.format(log,datetime.datetime.now(),psutil.virtual_memory().available / (1024**3),torch.cuda.memory_allocated() / (1024**3)))

        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        x_fast=tensor_reshape(self,fast,32)
        x_slow=tensor_reshape(self,slow,32)
        cpu_gpu_memory(self,'===tensor reshape===')
        batch_size = x_fast.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x_fast).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        cpu_gpu_memory(self,'===x chage to g===')
        if self.mode == "gaussian":
            theta_x = x_fast.view(batch_size, self.in_channels, -1)
            phi_x = x_fast.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_fast).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_fast).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)
            # cpu_gpu_memory(self,'===phi_x dot theta_x===')

        #in fun
        elif self.mode == "concatenate":
            theta_x = self.theta(x_fast).view(batch_size, self.inter_channels, -1, 1)
            cpu_gpu_memory(self,'===x chage to theta===')
            
            phi_x = self.phi(x_fast).view(batch_size, self.inter_channels, 1, -1)
            cpu_gpu_memory(self,'===x chage to phi===')
                        
            h = theta_x.size(2)
            w = phi_x.size(3)
            #cuda out of memory (64,64)
            theta_x = theta_x.repeat(1, 1, 1, w)
            cpu_gpu_memory(self,'===theta_x.repeat===')
            phi_x = phi_x.repeat(1, 1, h, 1)
            cpu_gpu_memory(self,'===phi_x.repeat===')

            #cuda out of memory (50,50)
            concat = torch.cat([theta_x, phi_x], dim=1)
            cpu_gpu_memory(self,'===theta_x & phi_x concat===')
            f = self.W_f(concat)
            cpu_gpu_memory(self,'===W_f===')
            f = f.view(f.size(0), f.size(2), f.size(3))
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
            cpu_gpu_memory(self,'softmax')
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y_fast = torch.matmul(f_div_C, g_x)
        cpu_gpu_memory(self,'===matmul(f,g)===')
        # contiguous here just allocates contiguous chunk of memory
        y_fast = y_fast.permute(0, 2, 1).contiguous()
        y_fast = y_fast.view(batch_size, self.inter_channels, *x_fast.size()[2:])
        
        W_y_fast = self.W_z(y_fast)
        # residual connection
        # z = W_y + x
        print(W_y_fast.shape)
        print(x_slow.shape)
        z=torch.mul(W_y_fast,x_slow)
        # reshape tensor
        z=tensor_reshape(self,z,64)
        cpu_gpu_memory(self,'===mul(fast,slow)===')

        return z


if __name__ == '__main__':
    import torch
    import pickle
    with open('/home/perry/Desktop/code/copy_code/animal_all/tools/experiment/x_slow_upsampled.pkl', 'rb') as f:
        x_slow = pickle.load(f)
    with open('/home/perry/Desktop/code/copy_code/animal_all/tools/experiment/x_fast_upsampled.pkl', 'rb') as f:
        x_fast = pickle.load(f)

    print('max CPU memory:{:.2f} GB\nmax GPU memory:{:.2f} GB'.format(psutil.virtual_memory().total / (1024**3),torch.cuda.get_device_properties(torch.device('cuda:0')).total_memory/ (1024**3)))

    start=datetime.datetime.now()
    net = NLBlockND(in_channels=80, mode='concatenate', dimension=3).cuda()
    print(net)
    #test 2 input
    
    out = net(x_fast,x_slow).cuda()
    end=datetime.datetime.now()
    duration = end - start
    print("total sec", duration.total_seconds())

    print(out.size())



