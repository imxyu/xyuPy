import os
import random

import imageio
import scipy.ndimage
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

from xyu.med.WL import *

'''
用于2D CNN
'''
class DataStorage_training(torch.utils.data.Dataset):
    def __init__(self, path_raw, path_targets, WC = 100, WW = 150, norm = False): # best for HCC tumor
        self.path_raw = path_raw
        self.path_targets = path_targets
        self.WC = WC
        self.WW = WW
        self.norm = norm
        self.list = os.listdir(self.path_targets)
        self.trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # 按照ImageNet初始化
        self.trans_totensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()

    def __len__(self):
        return len(self.list)

    def read_data(self, index):
        # DICOM Windowing and Converting
        raw_name, target_name = self.list[index], self.list[index]
        raw = pydicom.dcmread(os.path.join(self.path_raw, str(raw_name[:-3] + 'dcm')))
        raw_enhanced = WL(raw, self.WC, self.WW)
        raw_enhanced = torch.tensor(raw_enhanced)
        raw_PIL = self.toPIL(raw_enhanced.unsqueeze(0).byte())
        target = Image.open(os.path.join(self.path_targets, target_name))
        return raw_PIL, target

    def transform(self, raw, target):
        augmenter_raw = transforms.RandomApply([
            transforms.RandomApply([
                transforms.RandomAffine(0, scale=(0.75, 1.25))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine((0, 180))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine(0, shear=(-30, 30))
            ]),
            transforms.RandomApply([
                transforms.RandomVerticalFlip(0.5)
            ]),
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(0.5)
            ])
        ])

        augmenter_target = transforms.RandomApply([
            transforms.RandomApply([
                transforms.RandomAffine(0, scale=(0.75, 1.25))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine((0, 180))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine(0, shear=(-30, 30))
            ]),
            transforms.RandomApply([
                transforms.RandomVerticalFlip(0.5)
            ]),
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(0.5)
            ])
        ])

        seed = random.randint(0, 2**32)
        random.seed(seed)
        raw_aug = augmenter_raw(raw)
        raw_aug = self.trans_totensor(raw_aug)
        if self.norm:
            raw_aug = self.trans_normalize(raw_aug)
        random.seed(seed)
        target_aug = augmenter_target(target)
        target_aug = self.trans_totensor(target_aug)
        return raw_aug, target_aug  # Torch Tensor, Torch Tensor

    def __getitem__(self, index):
        x, y = self.read_data(index)
        x, y = self.transform(x, y)
        x = x.float()
        y[y!=0] = 1 # 强制生成0/1标签
        y = y.long()
        y = y.squeeze()
        return x, y


class DataStorage_validation(torch.utils.data.Dataset):
    def __init__(self, path_raw, path_targets, WC = 100, WW = 150, norm = False, num_some_of_valset=500): # best for HCC tumor
        self.path_raw = path_raw
        self.path_targets = path_targets
        self.WC = WC
        self.WW = WW
        self.norm = norm
        self.list = os.listdir(self.path_targets)
        self.num_some_of_valset = num_some_of_valset
        self.some_of_valset = self.__choose_part_of_valset__()
        self.trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]) # 按照ImageNet初始化
        self.trans_totensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()
    
    def __choose_part_of_valset__(self):
        if len(self.list) > self.num_some_of_valset:
            order = np.argsort(np.random.random(len(self.list)))
            part = np.array(self.list)[order][0:self.num_some_of_valset]
            return part.tolist()
        else:
            return self.list 

    def __len__(self):
        if len(self.list) >= self.num_some_of_valset:
            return self.num_some_of_valset
        else:
            return len(self.list)
    
    def read_data(self, index):
        # DICOM Windowing and Converting
        raw_name, target_name = self.some_of_valset[index], self.some_of_valset[index]
        raw = pydicom.dcmread(os.path.join(self.path_raw,str(raw_name[:-3] + 'dcm')))
        raw_enhanced = WL(raw, self.WC, self.WW)
        raw_enhanced = torch.tensor(raw_enhanced)
        raw_PIL = self.toPIL(raw_enhanced.unsqueeze(0).byte())
        target = Image.open(os.path.join(self.path_targets, target_name))
        return raw_PIL, target # return values are PIL Images

    def __getitem__(self, index):
        raw, target = self.read_data(index)
        raw = self.trans_totensor(raw)  # [0, 255] --> [0, 1]
        if self.norm:
            raw = self.trans_normalize(raw) # ImageNet Normalization
        target = self.trans_totensor(target)
        raw = raw.float()
        target[target != 0] = 1
        target = target.long()
        target = target.squeeze()
        return raw, target


class BMP_DataStorage_training(torch.utils.data.Dataset):
    def __init__(self, path_raw, path_targets, norm=False):  # best for HCC tumor
        self.path_raw = path_raw
        self.path_targets = path_targets
        self.norm = norm
        self.list = os.listdir(self.path_targets)
        self.trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])  # 按照ImageNet初始化
        self.trans_totensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()

    def __len__(self):
        return len(self.list)

    def read_data(self, index):
        # DICOM Windowing and Converting
        raw_name, target_name = self.list[index], self.list[index]
        raw = Image.open(os.path.join(self.path_raw, raw_name))
        target = Image.open(os.path.join(self.path_targets, target_name))
        return raw, target

    def transform(self, raw, target):
        augmenter_raw = transforms.RandomApply([
            transforms.RandomApply([
                transforms.RandomAffine(0, scale=(0.75, 1.25))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine((0, 180))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine(0, shear=(-30, 30))
            ]),
            transforms.RandomApply([
                transforms.RandomVerticalFlip(0.5)
            ]),
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(0.5)
            ])
        ])

        augmenter_target = transforms.RandomApply([
            transforms.RandomApply([
                transforms.RandomAffine(0, scale=(0.75, 1.25))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine((0, 180))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine(0, shear=(-30, 30))
            ]),
            transforms.RandomApply([
                transforms.RandomVerticalFlip(0.5)
            ]),
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(0.5)
            ])
        ])

        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        raw_aug = augmenter_raw(raw)
        raw_aug = self.trans_totensor(raw_aug)
        if self.norm:
            raw_aug = self.trans_normalize(raw_aug)
        random.seed(seed)
        target_aug = augmenter_target(target)
        target_aug = self.trans_totensor(target_aug)
        return raw_aug, target_aug  # Torch Tensor, Torch Tensor

    def __getitem__(self, index):
        x, y = self.read_data(index)
        x, y = self.transform(x, y)
        x = x.float()
        y[y != 0] = 1  # 强制生成0/1标签
        y = y.long()
        y = y.squeeze()
        return x, y


class BMP_DataStorage_validation(torch.utils.data.Dataset):
    def __init__(self, path_raw, path_targets, norm=False,
                 num_some_of_valset=500):  # best for HCC tumor
        self.path_raw = path_raw
        self.path_targets = path_targets
        self.norm = norm
        self.list = os.listdir(self.path_targets)
        self.num_some_of_valset = num_some_of_valset
        self.some_of_valset = self.__choose_part_of_valset__()
        self.trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])  # 按照ImageNet初始化
        self.trans_totensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()

    def __choose_part_of_valset__(self):
        if len(self.list) > self.num_some_of_valset:
            order = np.argsort(np.random.random(len(self.list)))
            part = np.array(self.list)[order][0:self.num_some_of_valset]
            return part.tolist()
        else:
            return self.list

    def __len__(self):
        if len(self.list) >= self.num_some_of_valset:
            return self.num_some_of_valset
        else:
            return len(self.list)

    def read_data(self, index):
        raw_name, target_name = self.list[index], self.list[index]
        raw = Image.open(os.path.join(self.path_raw, raw_name))
        target = Image.open(os.path.join(self.path_targets, target_name))
        return raw, target

    def __getitem__(self, index):
        raw, target = self.read_data(index)
        raw = self.trans_totensor(raw)  # [0, 255] --> [0, 1]
        if self.norm:
            raw = self.trans_normalize(raw)  # ImageNet Normalization
        target = self.trans_totensor(target)
        raw = raw.float()
        target[target != 0] = 1
        target = target.long()
        target = target.squeeze()
        return raw, target
'''
trans_normalize对输入的要求：
shape: [C, H, W]
type: torch.tensor
'''

class DICOM_DataStorage_training(DataStorage_training):
    pass

class DICOM_DataStorage_validation(DataStorage_validation):
    pass

class Multi_WL_DICOM_DataStorage_training(DataStorage_training):
    def __init__(self, path_raw, path_targets, norm = False): # best for HCC tumor
        self.path_raw = path_raw
        self.path_targets = path_targets
        self.norm = norm
        self.WL = ([40, 150], [25, 150], [50, 150], [75, 150], [100, 150], [125, 150], [150, 150],)
        self.list = os.listdir(self.path_targets)
        self.trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # 按照ImageNet初始化
        self.trans_totensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()

    def __len__(self):
        return len(self.list)

    def __read_data__(self, index):
        # DICOM Windowing and Converting
        raw_name, target_name = self.list[index], self.list[index]
        raw_group = ()
        raw = pydicom.dcmread(os.path.join(self.path_raw, str(raw_name[:-3] + 'dcm')))
        for config in self.WL:    
            raw_enhanced = WL(raw, config[0], config[1])
            raw_enhanced = torch.tensor(raw_enhanced)
            raw_PIL = self.toPIL(raw_enhanced.unsqueeze(0).byte())
            raw_group += (raw_PIL,)
        target = Image.open(os.path.join(self.path_targets, target_name))
        return raw_group, target

    def __transform__(self, raw, target, seed):
        '''
        使用了如下数据增强方法：
        缩放：0.75-1.25
        旋转：0-180 degree
        剪断shear：-30=30 degree
        纵向翻转：50%概率
        横向翻转：50%概率
        '''
        augmenter_raw = transforms.RandomApply([
            transforms.RandomApply([
                transforms.RandomAffine(0, scale=(0.75, 1.25))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine((0, 180))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine(0, shear=(-30, 30))
            ]),
            transforms.RandomApply([
                transforms.RandomVerticalFlip(0.5)
            ]),
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(0.5)
            ])
        ])

        augmenter_target = transforms.RandomApply([
            transforms.RandomApply([
                transforms.RandomAffine(0, scale=(0.75, 1.25))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine((0, 180))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine(0, shear=(-30, 30))
            ]),
            transforms.RandomApply([
                transforms.RandomVerticalFlip(0.5)
            ]),
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(0.5)
            ])
        ])

        random.seed(seed)
        raw_aug = augmenter_raw(raw)
        raw_aug = self.trans_totensor(raw_aug)
        if self.norm:
            raw_aug = self.trans_normalize(raw_aug)
        random.seed(seed)
        target_aug = augmenter_target(target)
        target_aug = self.trans_totensor(target_aug)
        return raw_aug, target_aug  # Torch Tensor, Torch Tensor

    def __getitem__(self, index):
        x = ()
        y = ()
        raw_group, target = self.__read_data__(index)
        seed = random.randint(0, 2**32)
        for img in raw_group:
            x_, y_ = self.__transform__(img, target, seed)
            x += (x_, )
            y += (y_, )
        x = torch.cat(x, 0)
        y = y_
        
        x = x.float()
        y[y!=0] = 1
        y = y.long()
        y = y.squeeze()
        return x, y

class Multi_WL_DICOM_DataStorage_validation(Multi_WL_DICOM_DataStorage_training):
    def __getitem__(self, index):
        x = ()
        y = ()
        raw_group, target = self.__read_data__(index)
        for img in raw_group:
            x_ = self.trans_totensor(img)
            x += (x_,)
        x = torch.cat(x, 0)
        y = self.trans_totensor(target)
        
        x = x.float()
        y[y!=0] = 1
        y = y.long()
        y = y.squeeze()
        return x, y

'''
用于随机选择窗宽窗位，训练阶段设计了6种窗宽窗位
验证阶段仅使用[100, 150]一种
'''
class Random_WL_DICOM_DataStorage_training(DataStorage_training):
    def __init__(self, path_raw, path_targets, norm = False): # best for HCC tumor
        self.path_raw = path_raw
        self.path_targets = path_targets
        # self.WL = ([0, 2048], [25, 150], [50, 150], [75, 150], [100, 150], [125, 150], [150, 150],)
        self.WL = ([0, 2000], [0, 1000], [0, 500], [0, 250], [75, 150], [100, 150], [125, 150])
        self.norm = norm
        self.list = os.listdir(self.path_targets)
        self.trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # 按照ImageNet初始化
        self.trans_totensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()

    def read_data(self, index):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        WL_idx = random.randint(0, 6)
        WC = self.WL[WL_idx][0]
        WW = self.WL[WL_idx][1]
        # DICOM Windowing and Converting
        raw_name, target_name = self.list[index], self.list[index]
        raw = pydicom.dcmread(os.path.join(self.path_raw,str(raw_name[:-3] + 'dcm')))
        raw_enhanced = WL(raw, WC, WW)
        raw_enhanced = torch.tensor(raw_enhanced)
        raw_PIL = self.toPIL(raw_enhanced.unsqueeze(0).byte())
        target = Image.open(os.path.join(self.path_targets, target_name))
        return raw_PIL, target # return values are PIL Images

# 验证过程只使用(100, 150)的图像作为输入
class Random_WL_DICOM_DataStorage_validation(DataStorage_validation):
    pass 

'''
用于2.5D CNN
将多张slices放至不同的channel作为网络输入
'''
class Multi_Silces_DataStorage_training(DataStorage_training):
    def __init__(self, path_raw, path_targets, WC = 100, WW = 150, N = 5, norm = False):
        self.path_raw = path_raw
        self.path_targets = path_targets
        self.WC = WC
        self.WW = WW
        self.N = N
        self.dist = int((self.N - 1) / 2)
        self.norm = norm
        self.list = os.listdir(self.path_targets)
        self.trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) # 按照ImageNet初始化
        self.trans_totensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()

    def __len__(self):
        return len(self.list)

    def __refresh_index__(self, index):
        # 判断左边界
        if (index - self.dist) < 0:
            index += abs(index - self.dist)
        # 判断右边界
        elif abs((len(self.list) - 1) - index) < self.dist:
            index -= (self.dist - ((len(self.list) - 1) - index))
        # print(index)
        return index 

    def __read_data__(self, index):
        raw_group = []
        for i in range(self.N):
            # print(len(self.list), index - self.dist + i)
            raw_name = self.list[index - self.dist + i]

            raw = pydicom.dcmread(os.path.join(self.path_raw, str(raw_name[:-3] + 'dcm')))
            raw_enhanced = WL(raw, self.WC, self.WW)
            raw_enhanced = torch.tensor(raw_enhanced)
            raw_PIL = self.toPIL(raw_enhanced.unsqueeze(0).byte())
            raw_group.append(raw_PIL)
        
        target_name = self.list[index]
        target = Image.open(os.path.join(self.path_targets, target_name))

        return raw_group, target # type: list, PIL Image

    def __transform__(self, raw, target, seed):
        '''
        使用了如下数据增强方法：
        缩放：0.75-1.25
        旋转：0-180 degree
        剪断shear：-30=30 degree
        纵向翻转：50%概率
        横向翻转：50%概率
        '''
        augmenter_raw = transforms.RandomApply([
            transforms.RandomApply([
                transforms.RandomAffine(0, scale=(0.75, 1.25))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine((0, 180))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine(0, shear=(-30, 30))
            ]),
            transforms.RandomApply([
                transforms.RandomVerticalFlip(0.5)
            ]),
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(0.5)
            ])
        ])

        augmenter_target = transforms.RandomApply([
            transforms.RandomApply([
                transforms.RandomAffine(0, scale=(0.75, 1.25))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine((0, 180))
            ]),
            transforms.RandomApply([
                transforms.RandomAffine(0, shear=(-30, 30))
            ]),
            transforms.RandomApply([
                transforms.RandomVerticalFlip(0.5)
            ]),
            transforms.RandomApply([
                transforms.RandomHorizontalFlip(0.5)
            ])
        ])

        random.seed(seed)
        raw_aug = augmenter_raw(raw)
        raw_aug = self.trans_totensor(raw_aug)
        if self.norm:
            raw_aug = self.trans_normalize(raw_aug)
        random.seed(seed)
        target_aug = augmenter_target(target)
        target_aug = self.trans_totensor(target_aug)
        return raw_aug, target_aug  # Torch Tensor, Torch Tensor

    def __getitem__(self, index):
        x = ()
        y = ()
        index = self.__refresh_index__(index)
        raw_group, target = self.__read_data__(index)
        seed = random.randint(0, 2**32)
        for i in range(self.N):
            x_, y_ = self.__transform__(raw_group[i], target, seed)
            x += (x_, )
            y += (y_, )
        x = torch.cat(x, 0)
        y = y[self.dist]
        
        x = x.float()
        # print(x.shape)

        y[y!=0] = 1
        y = y.long()
        y = y.squeeze()
        return x, y


class Multi_Silces_DataStorage_validation(Multi_Silces_DataStorage_training):
    def __getitem__(self, index):
        x = ()
        index = self.__refresh_index__(index)
        raw_group, target = self.__read_data__(index)
        for i in range(self.N):
            x_ = self.trans_totensor(raw_group[i])
            x += (x_,)
        x = torch.cat(x, 0)
        y = self.trans_totensor(target)

        x = x.float()
        # print(x.shape)

        y[y != 0] = 1
        y = y.long()
        y = y.squeeze()
        return x, y

'''
用于3D CNN
将某个patient的完整volume输入网络
'''


class VolumeDataStorage(torch.utils.data.Dataset):
    def __init__(self, path_raw, path_targets, is_half):
        self.path_raw = path_raw
        self.path_targets = path_targets
        self.is_half = is_half
        self.filenames = os.listdir(self.path_raw)
        self.raw = [np.load(os.path.join(self.path_raw, name)) for name in self.filenames]
        self.target = [np.load(os.path.join(self.path_targets, name)) for name in self.filenames]

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        raw = self.raw[index]
        target = self.target[index]
        target[target>1] = 1
        raw = torch.tensor(raw).unsqueeze(0).float() / 255.
        target = torch.tensor(target).long()
        # print(target.unique())
        if self.is_half:
            return raw.half(), target
        else:
            return raw, target


class VolumeDataStorage_from_disk(torch.utils.data.Dataset):
    def __init__(self, path_raw, path_targets, is_half=False):
        self.path_raw = path_raw
        self.path_targets = path_targets
        self.is_half = is_half
        self.filenames = os.listdir(self.path_raw)
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        raw = np.load(os.path.join(self.path_raw, self.filenames[index]))
        target = np.load(os.path.join(self.path_targets, self.filenames[index]))
        target[target>1] = 1
        raw = torch.tensor(raw).unsqueeze(0).float() / 255.
        target = torch.tensor(target).long()
        # print(target.unique())
        if self.is_half:
            return raw.half(), target
        else:
            return raw, target

class LITS_Random_WL_DICOM_DataStorage_training(DataStorage_training):
    def __init__(self, path_raw, path_targets, norm = False): # best for HCC tumor
        self.path_raw = path_raw
        self.path_targets = path_targets
        self.WL = ([0, 2048], [25, 150], [50, 150], [75, 150], [100, 150], [125, 150], [150, 150],)
        self.norm = norm
        self.list = os.listdir(self.path_targets)
        self.trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # 按照ImageNet初始化
        self.trans_totensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()

    def read_data(self, index):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        WL_idx = random.randint(0, 6)
        WC = self.WL[WL_idx][0]
        WW = self.WL[WL_idx][1]
        # DICOM Windowing and Converting
        raw_name, target_name = self.list[index], self.list[index]
        # raw = pydicom.dcmread(os.path.join(self.path_raw,str(raw_name[:-3] + 'dcm')))
        raw = np.load(os.path.join(self.path_raw, raw_name))
        raw_enhanced = WL_SLIVER07(raw, WC, WW)
        raw_enhanced = torch.tensor(raw_enhanced)
        raw_PIL = self.toPIL(raw_enhanced.unsqueeze(0).byte())
        # target = Image.open(os.path.join(self.path_targets, target_name))
        target = np.load(os.path.join(self.path_targets, target_name))
        target = torch.tensor(target).unsqueeze(0).byte()
        target = self.toPIL(target)
        return raw_PIL, target # return values are PIL Images


# 验证过程只使用(100, 150)的图像作为输入
class LITS_Random_WL_DICOM_DataStorage_validation(DataStorage_validation):
    def read_data(self, index):
        # DICOM Windowing and Converting
        raw_name, target_name = self.some_of_valset[index], self.some_of_valset[index]
        raw = np.load(os.path.join(self.path_raw, raw_name))
        raw_enhanced = WL_SLIVER07(raw, self.WC, self.WW)
        raw_enhanced = torch.tensor(raw_enhanced)
        raw_PIL = self.toPIL(raw_enhanced.unsqueeze(0).byte())
        target = np.load(os.path.join(self.path_targets, target_name))
        target = torch.tensor(target).unsqueeze(0).byte()
        target = self.toPIL(target)
        return raw_PIL, target # return values are PIL Images