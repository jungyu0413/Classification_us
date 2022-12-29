import glob
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np

class getdataset(Dataset):

    def __init__(self, path, type, ratio_0=1, ratio_1=1, ratio_2=1, ratio_3=1):
        self.path = path
        self.type = type

        if type == 'train':
            self.path_0 = path + '/train/0'
            self.path_1 = path + '/train/1'
            self.path_2 = path + '/train/2'
            self.path_3 = path + '/train/3'


            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
                transforms.RandomCrop(224),
           #     transforms.CenterCrop((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            #    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.Normalize([0.25429597, 0.25432256, 0.25452384], [0.03109195, 0.031083336, 0.030992588]),
                # 0.6
         #       transforms.Normalize([0.25135696, 0.25137797, 0.251608], [0.031570096, 0.031562865, 0.03147524]),
            ])   # 0.4

        elif type == 'valid':
            self.path_0 = path + '/valid/0'
            self.path_1 = path + '/valid/1'
            self.path_2 = path + '/valid/2'
            self.path_3 = path + '/valid/3'


            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
               # transforms.RandomCrop(224),
            #    transforms.CenterCrop((256, 128)),
                transforms.Normalize([0.25429597, 0.25432256, 0.25452384], [0.03109195, 0.031083336, 0.030992588]),
                # 0.6
            #    transforms.Normalize([0.25135696, 0.25137797, 0.251608], [0.031570096, 0.031562865, 0.03147524]),
            ])  # 0.4

        else:
            self.path_0 = path + '/test/0'
            self.path_1 = path + '/test/1'
            self.path_2 = path + '/test/2'
            self.path_3 = path + '/test/3'


            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
                #transforms.RandomCrop(224),
            #    transforms.CenterCrop((256, 128)),
                transforms.Normalize([0.25429597, 0.25432256, 0.25452384], [0.03109195, 0.031083336, 0.030992588]),
                # 0.6
            #    transforms.Normalize([0.25135696, 0.25137797, 0.251608], [0.031570096, 0.031562865, 0.03147524]),
            ])  # 0.4

        self.img_list_0 = glob.glob(self.path_0 + '/*.jpg')
        self.img_list_1 = glob.glob(self.path_1 + '/*.jpg')
        self.img_list_2 = glob.glob(self.path_2 + '/*.jpg')
        self.img_list_3 = glob.glob(self.path_3 + '/*.jpg')

        
        self.ratio_0 = ratio_0
        self.ratio_1 = ratio_1
        self.ratio_2 = ratio_2
        self.ratio_3 = ratio_3

        self.img_list = ratio_0*self.img_list_0 + ratio_1*self.img_list_1 + ratio_2*self.img_list_2 + ratio_3*self.img_list_3

        self.Image_list = []
        for img_path in self.img_list:
            self.Image_list.append(Image.open(img_path))

        self.class_list = [0] * len(ratio_0*self.img_list_0) + [1] * len(ratio_1*self.img_list_1) + [2] * len(ratio_2*self.img_list_2) + [3] * len(ratio_3*self.img_list_3) 


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        img = self.Image_list[idx]
        label = self.class_list[idx]
        img = self.transform(img)

        return img, label