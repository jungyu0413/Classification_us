import numpy as np
import torch.nn as nn
from utils import set_model_mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from imblearn.metrics import specificity_score
import matplotlib.pyplot as plt
import sys
import torch
from utils import grad_cam_view
import os

class_num = 4
def flatten(lst):
    result = []
    for item in lst:
        result.extend(item)
    return result

def get_counts(seq): 
    counts = {}
    for x in seq:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

def get_list(lst):
    result = []
    for i in lst:
        result.append(list(i))
    return result

def test_report(net, test_loader, save_name, name):
    pred_lt = []
    lb_lt = []
    score_lt = []
    net.cuda()
    set_model_mode('eval', [net])
#    net.load_state_dict(torch.load('/workspace/Pipeline/trained_models/net_{}_best.pt'.format(save_name)))
    save_folder = '/workspace/Pipeline_version3_manufacturer/2_test_report/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for data in test_loader:

        # 1. Source input -> Source Classification
        image, label = data
        image, label = image.cuda(), label.cuda().detach().cpu().numpy()
        output = net(image)

        score_val = output.detach().cpu().numpy()
        pred_val = output.data.max(1, keepdim=True)[1].flatten().detach().cpu().numpy()
        score_lt.append(list(score_val))
        pred_lt.append(list(pred_val))
        lb_lt.append(list(label))
    
    y_test = (flatten(lb_lt))
    preds = flatten(pred_lt)
    score = get_list(flatten(get_list(score_lt)))
    
    y_test_ohe = []
    for value in y_test:
        ohe = [0 for _ in range(class_num)]
        ohe[value] = 1
        y_test_ohe.append(ohe)

    preds_ohe = []
    for value in preds:
        ohe = [0 for _ in range(class_num)]
        ohe[value] = 1
        preds_ohe.append(ohe)


    sys.stdout = open(save_folder + '/{}.txt'.format(save_name), 'w')
    print('macro')
    df_confusion_margin = pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print('class weights:', get_counts(y_test))
    print(df_confusion_margin)
    print('accuracy_score:', accuracy_score(y_test, preds))
    print('balanced_accuracy_score:', balanced_accuracy_score(y_test, preds))
    print('recall_score:', recall_score(y_test, preds, average='macro'))
    print('PPV(precision_score):', precision_score(y_test, preds, average='macro'))
    print('Specificity:', specificity_score(y_test, preds, average='macro'))
    print('f1_score:', f1_score(y_test, preds, average='macro'))
    print("roc_auc_score: ", roc_auc_score(y_test_ohe, score, multi_class='ovr', average='macro'))


    print('micro')
    df_confusion_margin = pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print('class weights:', get_counts(y_test))
    print(df_confusion_margin)
    print('accuracy_score:', accuracy_score(y_test, preds))
    print('balanced_accuracy_score:', balanced_accuracy_score(y_test, preds))
    print('recall_score:', recall_score(y_test, preds, average='micro'))
    print('PPV(precision_score):', precision_score(y_test, preds, average='micro'))
    print('Specificity:', specificity_score(y_test, preds, average='micro'))
    print('f1_score:', f1_score(y_test, preds, average='micro'))
    print("roc_auc_score: ", roc_auc_score(y_test_ohe, score, multi_class='ovr', average='micro'))


    n_classes = 4
    # ROC & AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test_ohe = get_list(y_test_ohe)
    
    y_test_ohe = np.array(y_test_ohe)
    score = np.array(score)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_ohe[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    plt.figure(figsize=(20, 5))
    for idx, i in enumerate(range(n_classes)):
        plt.subplot(141+idx)
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Class %0.0f' % idx)
        plt.legend(loc="lower right")
    plt.savefig(save_folder + '/result_{}.png'.format(save_name), dpi=300)
    plt.show()
    # gradcam

    grad_img_url = ['/workspace/yolo_RoI_version4_manu(0.6)/train/0/19834904-4046.jpg',
                    '/workspace/yolo_RoI_version4_manu(0.6)/train/1/17734882-19037.jpg',
                    '/workspace/yolo_RoI_version4_manu(0.6)/train/2/21752252-36862.jpg',
                    '/workspace/yolo_RoI_version4_manu(0.6)/train/3/49570364-23632.jpg']

    for label, url in enumerate(grad_img_url):

        grad_cam_view(net, url, label, save_name, name)

    sys.stdout.close()  