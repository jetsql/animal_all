
import torch
import torch.nn as nn

x_slow=torch.randn(1,64,4,64,64)
x_fast_lateral=torch.randn(1,16,4,64,64)
import time

start=time.time()
print('=== original shape === \n{}\n{}'.format(x_slow.shape,x_fast_lateral.shape))
# 定義 1x1x1 的卷積層，將通道數從 16 提升為 80
conv_slow = nn.Conv3d(64, 80, kernel_size=1)
conv_fast = nn.Conv3d(16, 80, kernel_size=1)
# 通過卷積層進行維度提升
x_slow_upsampled = conv_slow(x_slow)
x_fast_upsampled = conv_fast(x_fast_lateral)
print('=== upsampeled shape === \n{}\n{}'.format(x_slow_upsampled.shape,x_fast_upsampled.shape))
# 進行 element-wise 相乘
x_slow = torch.mul(x_slow_upsampled,x_fast_upsampled)
print(x_slow.shape)
end=time.time()
print('==time== {}'.format(end - start))

#放到code裡面的（記得要加.cuda)
'''
        if self.slow_path.lateral:
            #######################33333#####################################
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)

            #定義 1x1x1 的卷積層，將通道數提升為 80
            conv_slow = nn.Conv3d(64, 80, kernel_size=1).cuda()
            conv_fast = nn.Conv3d(16, 80, kernel_size=1).cuda()
            # 通過卷積層進行維度提升
            x_slow_upsampled = conv_slow(x_slow).cuda()
            x_fast_upsampled = conv_fast(x_fast_lateral).cuda()
            # 進行 element-wise 相乘
            x_slow = torch.mul(x_slow_upsampled,x_fast_upsampled)
'''