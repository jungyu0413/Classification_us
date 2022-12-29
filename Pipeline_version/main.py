import torch
import train
import model
from utils import get_free_gpu
import utils
import torch.utils.data as torchdata
import get_dataset
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import os
import wandb
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="UltraSound Classification EXPERIMENTS")
parser.add_argument('-save_name', type=str)
parser.add_argument('-lr_sch', type=str)
# res, vgg, eff
parser.add_argument('-model', type=str)
parser.add_argument('-epochs', default=100, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-gpu', help='gpu number', type=str, default='0')


wandb.init(project="UltraSound-Classification", entity="jungyu0413", name='1227-res-Cyexp_0.6')


def main(args):
    try :
        os.makedirs('/workspace/Pipeline_version3_manufacturer/trained_models')
    except:
        pass

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.gpu
################# wandb #######################
    
    # can use wandb
    wandb.config = {
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    'model':args.model,
    'lr_sch':args.lr_sch
    }

################# dataset #####################
###############  upload dataset     ###########
###############################################
    trainset = get_dataset.getdataset(path='/workspace/yolo_RoI_version4_manu(0.6)', type='train')

################# dataloader ##################
    train_loader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)
   

    if torch.cuda.is_available():
        get_free_gpu()
        print('Running GPU : {}'.format(torch.cuda.current_device()))


################# Model #######################
        if args.model == 'res':
            net = nn.DataParallel(model.resnet50_build().cuda())
        elif args.model == 'vgg':
            net = nn.DataParallel(model.vgg16_bn().cuda())
        elif args.model == 'vgg19':
            net = nn.DataParallel(model.vgg19_bn().cuda())
        elif args.model == 'xcep':
            net = nn.DataParallel(model.Xception().cuda())
        elif args.model == 'vit':
            net = nn.DataParallel(model.ViT().cuda())
        else:
            print('This model is not here(model.py')

################# Train #######################
        train.fit(net, train_loader, args.save_name, args.epochs, args.batch_size, args.model)


    else:
        print("There is no GPU -_-!")


if __name__ == "__main__":
    
    args = parser.parse_args()
    main(args)