import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from psp_encoders.encoders import GradualStyleEncoder


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


imread = lambda x : cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
imsave = lambda path, x : cv2.imwrite(path, cv2.cvtColor(x, cv2.COLOR_RGB2BGR))

def to_torch(img):
    return torch.tensor(img.transpose((2, 0, 1))) / 255.

def from_torch(img):
    return (img.cpu().detach().numpy().transpose((1, 2, 0)) * 255).astype('uint8')


def unnormalize(img):
    img = ((img + 1) / 2)
    img[img < 0] = 0
    img[img > 1] = 1
    return img


def get_keys(ckpt, name):
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    return {k[len(name) + 1:]: v for k, v in ckpt.items() if k[:len(name)] == name}


def read_filepaths(dir, extension):
    with open(f'{data_dir}/datapoints.txt') as fin:
        ids = [id.strip('\n') for id in fin.readlines()]
    return [f'{data_dir}/{dir}/{id}.{extension}' for id in ids]


def batch(arrlist, batch_len):
    ret = []
    for i in range(0, len(arrlist), batch_len):
        ret.append(arrlist[i : i + batch_len])
    return ret


def read_img_batch(path_batch, transforms):
    imgs = list(
        map(transforms,
            map(to_torch,
                map(imread, path_batch)
            )
        )
    )
    return torch.stack(imgs).to(device).float()


data_dir = 'data/CelebAText-HQ'

img_paths = read_filepaths('images', 'jpg')
img_paths = batch(img_paths, batch_len=8)

"""
    StyleGAN used for debugging (comment out when done)
"""
from stylegan2.model import Generator

ckpt = torch.load('pretrained_models/stylegan2-ffhq-config-f.pt')

latent_avg = ckpt['latent_avg'].to(device)

stylegan = Generator(size=1024, style_dim=512, n_mlp=8)
stylegan.load_state_dict(ckpt['g_ema'])
stylegan.to(device)
stylegan.eval()

face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

print("StyleGAN Loaded")

"""
    Precompute w+ styles fed into stylegan
"""
ckpt = torch.load('pretrained_models/psp_ffhq_encode.pt')
ckpt['opts']['output_size'] = 1024
ckpt['opts']['n_styles'] = int(math.log(ckpt['opts']['output_size'], 2)) * 2 - 2

latent_encoder = GradualStyleEncoder(50, 'ir_se', ckpt['opts'])
latent_encoder.load_state_dict(get_keys(ckpt, 'encoder'))
latent_encoder.to(device)
latent_encoder.eval()

print("Latent Encoder Loaded")

latent_encoder_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)

import time

tic = time.time()
for i, path_batch in enumerate(img_paths):
    path_batch = ['c.jpg']
    if i % 10:
        print(i)
    batch = read_img_batch(path_batch, latent_encoder_transforms)
    batch = batch.to(device).float()
    print(batch)
    #print(latent_encoder)
    codes = latent_encoder(batch)
    print(codes)
    codes = codes + latent_avg.repeat(codes.shape[0], 1, 1)
    print(codes)
    imgs, _ = stylegan([codes], input_is_latent=True, randomize_noise=False)
    imgs = face_pool(imgs)
    print(imgs[0])
    imsave('b.jpg', from_torch(unnormalize(imgs[0])))
    exit()
    #img = imgs[0]
    #print(img.shape)
    #exit()
toc = time.time()
print(toc - tic)
