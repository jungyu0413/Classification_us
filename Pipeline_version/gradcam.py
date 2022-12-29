import torch
import model
import torch.nn as nn
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image,preprocess_image


def grad_cam_view(url, label, net, name):
    
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
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        concat_images = np.hstack((np.uint8(255*img), cam , cam_image))
        save_img = Image.fromarray(concat_images)
        save_path = '/workspace/Pipeline/3_gradcam/' + str(label) + '/.jgp'
        save_img.save(save_path)
    else:
        pass