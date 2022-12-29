from cmath import exp
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Function
from sklearn.manifold import TSNE
import torch
import itertools
import os
import torch.utils.data as torchdata
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



def optimizer_scheduler(epoch, optimizer):
    for param_group in optimizer.param_groups:
       # if epoch<= 10:
        #    pass
        #else:
        #    param_group['weight_decay'] = 0.001
           param_group['weight_decay'] = 1e-4
    return optimizer




def save_model(net, save_name, cs_name):
    print('Save models ...')

    save_folder = 'trained_models/'  + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(net.state_dict(), save_folder + '/net_' + str(save_name) + str(cs_name) + '.pt')

    print('Model is saved !!!')


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    
    return memory_available
    #return np.argmax(memory_available)

def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()




def BestSave(net, test_acc, save_name):
    try:
        if test_acc[-1] > max(test_acc[:-1]):
            save_model(net, save_name, '_best')
            print('save best!', 'best : ', test_acc[-1], 'before : ', max(test_acc[:-1]))
        else:
            print('not best!', 'best : ', max(test_acc[:-1]), 'now : ', test_acc[-1])

    except:
        print('early!')



class EarlyStopping:
    def __init__(self, patience=10):
        self.test_roc_score = 0.0
        self.patience = 0
        self.patience_limit = patience
        
    def step(self, test_roc_score):
        if self.test_roc_score < test_roc_score:
            self.test_roc_score = test_roc_score
            self.patience = 0
        else:
            self.patience += 1
    
    def is_stop(self):
        return self.patience >= self.patience_limit

    def is_num(self):
        return self.patience, self.test_roc_score

import torch
import model
import torch.nn as nn
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image,preprocess_image


def grad_cam_view(net, url, label, save_name ,name):
    
    net.cuda()
    set_model_mode('eval', [net])
    # image sample
    img = np.array(Image.open(url))

    image_resize = cv2.resize(img, (224, 224))

    image_scale = np.float32(image_resize) / 255

    input_tensor = preprocess_image(image_scale, mean=[-0.055694155, -0.055332225, -0.055633955], std=[0.99793416, 0.9979854, 0.9977238])
    
    if name == 'res':
        target_layers = [net.module.encoder.layer4]
        targets = [ClassifierOutputTarget(label)]
        cam = GradCAM(model=net, target_layers=target_layers)
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(image_scale, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        concat_images = np.hstack((np.uint8(255*image_scale), cam , cam_image))
        save_img = Image.fromarray(concat_images)
        save_path = '/workspace/Pipeline_version3_manufacturer/3_gradcam/' + save_name +'/' + str(label) + '.jpg'
        try:
            os.mkdir('/workspace/Pipeline_version3_manufacturer/3_gradcam/' + save_name)
        except:
            pass
        save_img.save(save_path)
    
    elif name == 'vgg':
        target_layers = [net.module.features[42]]
        targets = [ClassifierOutputTarget(label)]
        cam = GradCAM(model=net, target_layers=target_layers)
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(image_scale, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        concat_images = np.hstack((np.uint8(255*image_scale), cam , cam_image))
        save_img = Image.fromarray(concat_images)
        save_path = '/workspace/Pipeline_version3_manufacturer/3_gradcam/' + save_name +'/' + str(label) + '.jpg'
        try:
            os.mkdir('/workspace/Pipeline_version3_manufacturer/3_gradcam/' + save_name)
        except:
            pass
        save_img.save(save_path)
    
    elif name == 'vgg19':
        target_layers = [net.module.features[51]]
        targets = [ClassifierOutputTarget(label)]
        cam = GradCAM(model=net, target_layers=target_layers)
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(image_scale, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        concat_images = np.hstack((np.uint8(255*image_scale), cam , cam_image))
        save_img = Image.fromarray(concat_images)
        save_path = '/workspace/Pipeline_version3_manufacturer/3_gradcam/' + save_name +'/' + str(label) + '.jpg'
        try:
            os.mkdir('/workspace/Pipeline_version3_manufacturer/3_gradcam/' + save_name)
        except:
            pass
        save_img.save(save_path)

    else:
        target_layers = [net.module[1][11][1].fn[0]]
        targets = [ClassifierOutputTarget(label)]
        cam = GradCAM(model=net, target_layers=target_layers)
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(image_scale, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        concat_images = np.hstack((np.uint8(255*image_scale), cam , cam_image))
        save_img = Image.fromarray(concat_images)
        save_path = '/workspace/Pipeline_version3_manufacturer/3_gradcam/' + save_name +'/' + str(label) + '.jpg'
        try:
            os.mkdir('/workspace/Pipeline_version3_manufacturer/3_gradcam/' + save_name)
        except:
            pass
        save_img.save(save_path)