#實驗放在/home/perry/Desktop/code/copy_code/animal_all/mmaction/models/backbones/resnet3d_slowfast.py
#o_resnet3d_slowfast.py是原版
#resnet3d_slowfast.py是實驗版
import torch
import torch.nn as nn
x_slow=torch.randn(1,64,4,64,64)
x_fast_lateral=torch.randn(1,16,4,64,64)
import time

start=time.time()
# print('===== orange shap =====')
# print('slow ==> {}\nfast ==> {}'.format(x_slow.shape,x_fast_lateral.shape))
x_fast_repeat=x_fast_lateral.repeat(1,4,1,1,1)
# print('===== x_fast_repeat result =====')
# print(x_fast_repeat.shape)
# print('===== element wise =====')
x_slow=torch.mul(x_slow,x_fast_repeat)
# print(x_slow.shape)
conv_slow = nn.Conv3d(64, 80, kernel_size=1).cuda()
x_slow= conv_slow(x_slow).cuda()
# print(x_slow.shape)

end=time.time()
print('==time== {}'.format(end - start))