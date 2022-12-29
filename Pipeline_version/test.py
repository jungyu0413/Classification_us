import torch
import numpy as np
from utils import set_model_mode
import torch.nn.functional as F
import torch.nn as nn
from utils import EarlyStopping
from utils import BestSave
from utils import save_model
import os
import wandb
import torch.utils.data as torchdata
import get_dataset
from sklearn.metrics import roc_auc_score

test_acc_lt = []
test_loss_lt = []
class_num = 4
early_stop = EarlyStopping(patience=10)
def tester(net, test_loader, save_name, batch_size, epoch) :
    print("Model test ...")

    # binary
    #classifier_criterion = nn.BCEWithLogitsLoss()
    # multi
    classifier_criterion = nn.CrossEntropyLoss().cuda()
    net.cuda()
    set_model_mode('eval', [net])
    
    correct = 0 
    score_lt = []
    lb_lt = []
    for data in test_loader:
        count = len(test_loader)

        # input 
        image, label = data
        image, label = image.cuda(), label.cuda()
        output = net(image)
        score_val = output.detach().cpu().numpy()
        score_lt.append(list(score_val))
        lb_lt.append(list(label))

        # multi
        test_class_loss = classifier_criterion(output, label)
        # binary
        #test_class_loss = classifier_criterion(output.squeeze(), label.float())
      #  print('label:', label.float())
        # binary
        #pred = []
        #for dt in output:
         #   if dt > 0:
          #      pred.append(1)
           # else:
            #    pred.append(0)

        y_test = (flatten(lb_lt))
        score = get_list(flatten(get_list(score_lt)))

        y_test_ohe = []
        for value in y_test:
            ohe = [0 for _ in range(class_num)]
            ohe[value] = 1
            y_test_ohe.append(ohe)

        # multi
        pred = output.data.max(1, keepdim=True)[1]
     #   print('pred :', pred)
        # max tensor index
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    test_roc_score = roc_auc_score(y_test_ohe, score, multi_class='ovr', average='macro')
    

    print("\nTest results on training :")
    print('\nAccuracy: {}/{} ({:.2f}%)\n'
              .format(
            correct, (count*batch_size), 100. * correct.item() / (count*batch_size),
            ))
    print('roc_auc_score :', test_roc_score)
    test_acc = 100*(correct.item() / (count*batch_size))
    test_loss = test_class_loss.detach().cpu()
    test_acc_lt.append(test_acc)
#    test_loss_lt.append(test_loss)
#    test_roc_score_lt.append(test_roc_score)

    BestSave(net, test_acc_lt, save_name)
    early_stop.step(test_roc_score)
    print('patience :', early_stop.is_num())
    stop_bool = early_stop.is_stop()
    # ...

    #np.save('/workspace/Pipeline/trained_models/test_acc_lt-{}'.format(save_name), test_acc_lt)
    #np.save('/workspace/Pipeline/trained_models/test_loss_lt-{}'.format(save_name), test_loss_lt)
    return test_acc, test_loss, stop_bool


def flatten(lst):
  result = []
  for item in lst:
      result.extend(item)
  return result


def get_list(lst):
    result = []
    for i in lst:
        result.append(list(i))
    return result