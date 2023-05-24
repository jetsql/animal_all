#實驗放在/home/perry/Desktop/code/copy_code/animal_all/mmaction/models/backbones/resnet3d_slowfast.py
#o_resnet3d_slowfast.py是原版
#resnet3d_slowfast.py是實驗版
import torch
x_slow=torch.randn(1,64,4,64,64)
x_fast_lateral=torch.randn(1,16,4,64,64)
import time

start=time.time()

#val 2 tensor shape are same?
if torch.equal(x_slow, x_fast_lateral):
    x_slow_wise=torch.mul(x_slow,x_fast_lateral)
else:
    #separate the tensor part to slow and fast ([x,x,4, 64, 64])
    fast_temp = x_fast_lateral[0, 0, :, :, :].squeeze(dim=0)
    slow_temp = x_slow[0, 0, :, :, :].squeeze(dim=0)
    # print('===== temp size =====')
    # print('fast_temp ==>{}'.format(fast_temp.shape))
    # print('slow_temp ==>{}'.format(slow_temp.shape))
    # print('===== element wise =====')
    #splicing the previously processed tensor
    element_wise = torch.mul(fast_temp,slow_temp)
    print(element_wise.shape)


#這邊要加non-local
x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
# print('===== cat result =====')
# print(x_slow.shape)
# print('===== conneted (4,64,64)=====')
x_slow[0, 0, :, :, :]=element_wise.unsqueeze(dim=0)
print(x_slow.shape)
end=time.time()
print('==time== {}'.format(end - start))

'''放置在resnet3d_slowfast版本
        if self.slow_path.lateral:
            # x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
            # x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
            #####################################################
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)

            #copy tensor
            # x_slow_wise=x_slow.detach()
            #val 2 tensor shape are same?
            if torch.equal(x_slow, x_fast_lateral):
                x_slow=torch.mul(x_slow,x_fast_lateral)
            else:
                #separate the tensor part to slow and fast ([x,x,4, 64, 64])
                fast_temp = x_fast_lateral[0, 0, :, :, :].squeeze(dim=0)
                slow_temp = x_slow[0, 0, :, :, :].squeeze(dim=0)
                print('===== temp size =====')
                print('fast_temp ==>{}'.format(fast_temp.shape))
                print('slow_temp ==>{}'.format(slow_temp.shape))
                print('===== element wise =====')
                #splicing the previously processed tensor
                element_wise = torch.mul(fast_temp,slow_temp)
                print(element_wise.shape)
                #這邊要加non-local
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
                print('===== cat result =====')
                print(x_slow.shape)
                print('===== conneted (4,64,64)=====')
                x_slow[0, 0, :, :, :]=element_wise.unsqueeze(dim=0)
                print(x_slow.shape)
'''