from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class my_data(Dataset):
    {
#    def __init__(self, image_path, annotation_path, transform = None):
        # 初始化，读取数据集

#    def __len__(self):
        # 获取数据集总大小

#    def __getitem__(self, id):
            # 对于指定的Id，读取该数据并返回

    }

dataset = my_data("your image path", "your annotation path")    # 实例化该类
for data in dataset:
    print(data)

# 将transforms集成到Dataset类中，使用Compose将多个变换整合到一起
dataset = my_data("your image path", "your annotation path", transforms = transforms.Compose([
                  transforms.Resize(256),   # 将图像最短边缩小至256，宽高比例不变
                  # 以0.5的概率随机翻转指定的PIL图像
                  transforms.RandomHorizontalFlip(),
                  # 将PIL图像转为Tensor，元素区间从[0, 255]归一到[0, 1]
                  transforms.ToTensor(),
                  # 进行mean与std为0.5的标准化
                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]))

# 使用Dataloader进一步封装Dataset
dataloader = DataLoader(dataset, batch_size = 4, shuffle = True, num_workers = 4)
# 封装成迭代器
data_iter = iter(dataloader)
for step in range(iters_per_epoch):
    data = next(data_iter)
    # 将data用于训练网络即可
