import time
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
    return torch.clamp((img + 1) / 2, 0, 1)


def get_keys(ckpt, name):
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    return {k[len(name) + 1:]: v for k, v in ckpt.items() if k[:len(name)] == name}


def read_filepaths(dir, extension):
    with open(f'{data_dir}/datapoints.txt') as fin:
        ids = [id.strip('\n') for id in fin.readlines()]
    return [f'{data_dir}/{dir}/{id}.{extension}' for id in ids]


def batch_arrlist(arrlist, batch_len):
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
    Precompute w+ styles
"""
def load_ckpt(path):
    ckpt = torch.load(path)
    ckpt['opts']['output_size'] = 1024
    ckpt['opts']['n_styles'] = int(math.log(ckpt['opts']['output_size'], 2)) * 2 - 2

    model = GradualStyleEncoder(50, 'ir_se', ckpt['opts'])
    model.load_state_dict(get_keys(ckpt, 'encoder'))
    model.to(device)
    model.eval()

    return model


def generate_codes(paths, encoder, encoder_transforms, grayscale=False):
    paths_batched = batch_arrlist(paths, batch_len=8)
    all_codes = []

    for i, path_batch in enumerate(paths_batched):
        if i % 25 == 0:
            print(i)
        batch = read_img_batch(path_batch, encoder_transforms)
        if grayscale:
            batch = torch.unsqueeze(batch[:, 0, ...], dim=1)
        batch = batch.to(device).float()
        codes = encoder(batch)
        codes = codes + latent_avg.repeat(codes.shape[0], 1, 1)
        all_codes.extend([c for c in codes.cpu().detach()])

        # Uncomment to verify w/ reconstruction
        #imgs, _ = stylegan([codes], input_is_latent=True, randomize_noise=False)
        #imgs = face_pool(imgs)

        #imsave('a.jpg', imread(path_batch[0]))
        #imsave('b.jpg', from_torch(unnormalize(imgs[0])))

    return torch.stack(all_codes)


'''
img_encoder = load_ckpt('pretrained_models/psp_ffhq_encode.pt')
print("Image Encoder Loaded")

img_encoder_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

img_paths = read_filepaths('images', 'jpg')
img_codes = generate_codes(img_paths, img_encoder, img_encoder_transforms)
torch.save(img_codes, f'{data_dir}/img_codes.pt')
'''

img_codes = torch.load(f'{data_dir}/img_codes.pt')

sketch_encoder = load_ckpt('pretrained_models/psp_celebs_sketch_to_face.pt')
print("Sketch Encoder Loaded")

sketch_encoder_transforms = transforms.Compose([
    transforms.Resize((256, 256))
])

sketch_paths = read_filepaths('sketches', 'jpg')
sketch_codes = generate_codes(sketch_paths, sketch_encoder, sketch_encoder_transforms, grayscale=True)
torch.save(sketch_codes, f'{data_dir}/sketch_codes.pt')
