
'''
def to_torch(img):
    return torch.tensor(img.transpose((2, 0, 1)))

def from_torch(img):
    return img.detach().numpy().transpose((1, 2, 0))
'''
import numpy as np
import cv2
import torch

def get_keys(ckpt, name):
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    return {k[len(name) + 1:]: v for k, v in ckpt.items() if k[:len(name)] == name}


x = torch.load('pretrained_models/psp_ffhq_encode.pt')
