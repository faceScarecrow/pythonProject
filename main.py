# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

#import torch
#from perception import Perception           # 调用上述模块

#import torch
# 从上述文件中引入Perception类
#from perception_sequential import Perception


# 实例化一个网络，并赋值全连接中的维数，最终输出二维代表了二分裂
#perception = Perception(2, 3, 2)
# 可以看到perception中包含perception.py中定义的layer1与layer2
#perception
# named_parameters()可以返回学习参数的迭代器，分别为参数名与参数值
#for name, parameter in perception.named_parameters():
# 随机生成数据，注意这里的4代表了样本数为4，每个样本有两维
#data = torch.randn(4, 2)
#data
# 将输入数据传入perception，perception()相当于调用perception中的forward()函数
#output = perception(data)
#output

#model = Perception(100, 1000, 10).cuda()    # 构建类的实例，并表明在CUDA上
# 打印model结构，会显示Sequential中每一层的具体参数配置
#model
#input = torch.randn(100).cuda()
#output = model(input)                       # 将输入传入实例化的模型
#output.shape

# 接着2.3.1.节中的终端环境继续运行，来进一步求损失
#from torch import nn
#import torch.nn.functional as F
# 设置标签，由于是二分类，一共有4个样本，因此标签维度为1×4，每个数为0或1两个类别
#label = torch.Tensor([0, 1, 1, 0]).long()
# 实例化nn中的交叉熵损失类
#criterion = nn.CrossEntropyLoss()
# 调用交叉熵损失
#loss_nn = criterion(output, label)
#loss_nn
# 由于F.cross_entropy是一个函数，因此可以直接调用，不需要实例化，两者求得的损失值相同
#loss_functional = F.cross_entropy(output, label)
#loss_loss_functional

# Pytorch常用优化器
#from torch import optim
#optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
#optimizer = optim.Adam([var1, var2], lr = 0.0001)

#import torch
#from mlp import MLP
#from torch import optim
#from torch import nn
# 实例化模型，并赋予每一层的维度
#model = MLP(28 * 28, 300, 200, 10)
#model                                      # 打印model的结构，由3个全连接层组成
# 采用SGD优化器，学习率为0.01
#optimizer = optim.SGD(params = model.parameters(), lr = 0.01)
#data = torch.randn(10, 28 * 28)
#output = model(data)
# 由于是10分类，因此label元素从0到9，一共10个样本
#label = torch.Tensor([1, 0, 4, 7, 9, 3, 4, 5, 3, 2]).long()
#label
#criterion = nn.CrossEntropyLoss
#loss = criterion(output, label)
#loss
# 清空梯度，在每次优化前都需要进行此操作
#optimizer.zero_grad()
# 损失的反向传播
#loss.backward()
# 利用优化器进行梯度更新
#optimizer.step()

# 对于model中需要单独赋予学习率的层，如special层，则使用‘lr’关键字单独赋予
#optimizer = optim.SGD(
#    [{'params': model.special.parameters(), 'lr': 0.001}，
#    {'params': model.base.parameters()}, lr = 0.0001)

#from torch import nn
#from torchvision import models
# 通过torchvision.model直接调用VGG16的网络结构
#vgg = model.vgg16()
# VGG16的特征层包括13个卷积、13个激活函数ReLU、5个池化，一共31层
#len(vgg.features)
# VGG16的分类层包括3个全连接、2个ReLU、2个Dropout，一共7层
#len(vgg.classifier)
# 可以通过出现的顺序直接索引每一层
#vgg.classifier[-1]
# 也可以选取某一部分，如下代表了特征网络的最后一个卷积模组
#vgg.features[24:]

# 加载预训练模型
# 直接利用torchvision.models中自带的预训练模型
#from torch import nn
#from torchvision import models
# 通过torchvision.model直接调用VGG16的网络结构
#vgg = models.vgg16(pretrained = True)

# 使用自己的本地预训练模型，或者之前训练过的模型
#import torch
#from torch import nn
#from torchvision import models
# 通过torchvision.model直接调用VGG16的网络结构
#vgg = models.vgg16()
#state_dict = torch.load("your model path“)
# 利用load_state_dict,遍历预训练模型的关键字，如果出现在了VGG中，则加载预训练参数
#vgg.load_state_dict({k:v for k, v in state_dict_items() if k in vgg.state_dict()})

# GPU加速
#import torch
#from torchvision import models
#a = torch.randn(3, 3)
#b = models.vgg16()
# 判断当前GPU是否可用
#if torch.cuda.is_available():
           #a = a.cuda()
           # 指定将b转移到编号为1的GPU上
           #b = b.cuda(1)
# 使用torch.device()来指定使用哪一个GPU
#device = torch.device("cuda: 1")
#c = torch.randn(3, 3, device = device, requires_grad = True)

# 卷积
#from torch import nn
# 使用torch.nn中的Conv2d()搭建卷积层
#conv = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1,bias = True)
# 查看卷积核的基本信息，本质上是一个Module
#conv
# 通过.weight与.bias查看卷积核的权重与偏置
#conv.weight.shape
#conv.bias.shape
# 输入特征图，需要注意特征必须是四维，第一维作为batch数，即使是1也要保留
#input = torch.ones(1, 1, 5, 5)
#output = conv(input)
# 当前配置的卷积核可以使输入和输出的大小一致
#input.shape
#output.shape

# 激活函数层
# Sigmoid函数
# 引入torch.nn模块
#import torch
#from torch import nn
#input = torch.ones(1, 1, 2, 2)
#input
#sigmoid = nn.Sigmoid()                         # 使用nn.Sigmoid()实例化sigmoid
#sigmoid(input)

# ReLU函数
#import torch
#from torch import nn
#input = torch.randn(1, 1, 2, 2)
#input
# nn.ReLU()可以实现inplace操作，即可以直接将运算结果覆盖到输入中，以节省内存
#relu = nn.ReLU(inplace = True)
#relu(input)                                    # 可以看出大于0的值保持不变，小于0的值被置为0

# Leaky ReLU函数
#import torch
#from torch import nn
#input = torch.randn(1, 1, 2, 2)
#input
# 利用nn.LeakyReLU()构建激活函数，并且其1/ai为0.04，即ai为25，True代表in-place操作
#leakyrelu = nn.LeakyReLU(0.04, True)
#leakyrelu(input)                               # 从结果看大于0的值保持不变，小于0的值被以0.04的比例缩小

# Softmax函数
#import torch.nn.functional as F
#score = torch.randn(1, 4)
#score
# 利用torch.nn.functional.softmax()函数，第二个参数表示按照第几个维度进行
# Softmax计算
# F.softmax(score, 1)


# 池化层
#import torch
#from torch import nn
# 池化主要需要两个参数，第一个参数代表池化区域大小，第二个参数表示步长
#max_pooling = nn.MaxPool2d(2, stride = 2)
#aver_pooling = nn.AvgPool2d(2, stride = 2)
#input = torch.randn(1, 1, 4, 4)
#input
# 调用最大值池化与平均值池化，可以看到size从[1, 1, 4, 4]变成了[1, 1, 2, 2]
#max_pooling(input)
#aver_pooling(input)

# Dropout层
#import torch
#from torch import nn
# PyTorch将元素置0来实现Dropout层，第一个参数为置0概率，第二个为是否原地操作
#dropout = nn.Dropout(0.5, inplace = False)
#input = torch.randn(2, 64, 7, 7)
#output = dropout(input)

# BN层
#from torch import nn
# 使用BN层需要传入一个参数为num_features，即特征的通道数
#bn = nn.BatchNorm2d(64)
# eps为公式中的ε，momentum为均值方差的动量，affine为添加可学习参数
#bn
#input = torch.randn(4, 64, 224, 224)
#output = bn(input)
# BN层不改变输入、输出的特征大小
#output.shape

# 全连接层
#import torch
#from torch import nn
# 第一维表示一共有4个样本
#input = torch.randn(4, 1024)
#linear = nn.Linear(1024, 4096)
#output = linear(input)
#input.shape
#output.shape

# 空洞卷积
#from torch import nn
# 定义普通卷积，默认dilation为1
#conv1 = nn.Conv2d(3, 256, 3, stride = 1, padding = 1, dilation = 1)
#conv1
# 定义dilation为2的卷积，打印卷积后会有dilation的参数
#conv2 = nn.Conv2d(3, 256, 3, stride = 1, padding = 1, dilation = 2)
#conv2


# VGGNet
#import torch
#from vgg import VGG
# 实例化VGG类，在此设置输出分类树为21，并转移到GPU上
#vgg = VGG(21).cuda()
#input = torch.randn(1, 3, 224, 224).cuda()
#input.shape
# 调用VGG，输出21类的得分
#scores = vgg(input)
#scores.shape
# 也可以单独调用卷积模块，输出最后一层的特征图
#features = vgg.features(input)
#features.shape
# 打印出VGGNet的卷积层，5个卷积组一共30层
#vgg.features
# 打印出VGGNet的3个全连接层
#vgg.classifier


# Inception
import torch
from inceptionv1 import Inceptionv1
# 网络实例化，输入模块通道数，并转移到GPU上
net_inceptionv1 = Inceptionv1(3, 64, 32, 64, 64, 96, 32).cuda()
net_inceptionv1
input = torch.randn(1, 3, 256, 256).cuda()
input.shape
output = net_inceptionv1(input)
# 可以看到输出的通道数是输入通道数的和，即256 = 64 + 64 + 96 + 32
output.shape









def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

a = torch.Tensor(2, 2)


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
