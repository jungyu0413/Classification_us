import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
from utils import save_model
from utils import set_model_mode
import torch.utils.data as torchdata
import get_dataset
import torch.nn.functional as F
import torch.hub
import wandb
from utils import optimizer_scheduler
import torch.utils.model_zoo as model_zoo
import real_test
from utils import CosineAnnealingWarmUpRestarts
from utils import EarlyStopping

#test_lt = []
################# Train #######################
###############   ResNet     ##################
###############################################

def fit(net, train_loader, save_name, epochs, batch_size, name):
    validset = get_dataset.getdataset(path='/workspace/yolo_RoI_version4_manu(0.6)', type='valid')
    test_loader = torchdata.DataLoader(validset, batch_size, shuffle=True, pin_memory=True, drop_last=False)
    real_testset = get_dataset.getdataset(path='/workspace/yolo_RoI_version4_manu(0.6)', type='test')
    real_testloader = torchdata.DataLoader(real_testset, batch_size=32, shuffle=True, pin_memory=True, drop_last=False)

    train_loss_lt = []
    train_acc_lt = []
    print("fit training")


################# loss ####################`###
###############   optim     ##################
##############################################

    class_weights = torch.FloatTensor([ 0.45204842,  0.7446319 ,  2.52864583, 20.22916667]).cuda()

 

    # weighted CEL
    classifier_criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    # CEL
#    classifier_criterion = nn.CrossEntropyLoss().cuda()
    # BCE
    #classifier_criterion = nn.BCEWithLogitsLoss()
    
    # define loss function
    # Adam
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    # SGD
    optimizer = optim.SGD(
        list(net.parameters()),
        lr=0.0001, momentum=0.9)

    # OneCycle
   # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.015, 
    #                                         steps_per_epoch=25, epochs=20 ,anneal_strategy='linear')
    # Cycle
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, 
                                              step_size_up=5, max_lr=0.015, 
                                              gamma=0.5, mode='exp_range')
    
    #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=0.0015,  T_up=5, gamma=0.5)


    # Cycle
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.005,step_size_up=15,mode="exp_range")

    # expoential
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    for epoch in range(epochs):
      # total epoch
        print('Epoch : {}'.format(epoch))
        set_model_mode('train', [net])

################# data #######################
###############   image, label     ###########
############################################## 
        correct = 0
        for batch_idx, data in enumerate(train_loader):
            count = len(train_loader)
            image, label = data
            image, label = image.cuda(), label.cuda()
            # zero
            optimizer.zero_grad()
            #optimizer = utils.optimizer_scheduler(epoch=epoch, optimizer=optimizer)
################# Model ######################
###############    Input    ##################
##############################################
            class_pred = net(image)
            # feature extractor output

          
          #  print('pred : ', class_pred)
          #  print('label : ', label)

           # print('pred shape : ', class_pred.shape)
           # print('label shape: ', label.shape)

            # classification output
            # multi
            class_loss = classifier_criterion(class_pred, label)
            # binary
            #class_loss = classifier_criterion(class_pred.squeeze(), label.float())
            
            # backpropagation
            class_loss.backward()

            # focal loss
        #    class_loss_it = classifier_criterion(class_pred, label)
        #    pt = torch.exp(-class_loss_it)
        #    class_loss = (0.25 * (1-pt)**2 * class_loss_it).mean()

            # prediction
            pred = class_pred.data.max(1, keepdim=True)[1]
            # count
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            
            # optimizer
            optimizer.step()

            

            # input lr
            for param_group in optimizer.param_groups:
                optim_val = param_group['lr']
            

            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(image), len(train_loader.dataset), 100. * batch_idx / len(train_loader), class_loss.item()))
            # custom print
            train_loss_lt.append(class_loss.detach().cpu())
            train_acc = 100. * correct.item() / (count*batch_size)
            train_acc_lt.append(train_acc)

        
        scheduler.step()


        test_acc, test_loss, stop_bool= test.tester(net, test_loader ,save_name, batch_size, epoch)
        wandb.log({"train_accuracy":train_acc, "tune_accuracy":test_acc, "train_loss":class_loss, "tune_loss":test_loss, "optimizer":optim_val}, step=epoch)
        if stop_bool:
            print("Early stopping")
            break
            
        else:
            pass

    save_model(net, save_name, '_last')
    real_test.test_report(net, real_testloader, save_name, name)

    #np.save('/workspace/Pipeline/trained_models/train_loss_lt-{}'.format(save_name), train_loss_lt)
    #np.save('/workspace/Pipeline/trained_models/train_acc_lt-{}'.format(save_name), train_acc_lt)