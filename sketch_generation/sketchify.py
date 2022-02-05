import numpy as np
import cv2
import torch
from model import model, immean, imstd


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

model.load_state_dict(torch.load('../pretrained_models/sketchify_model_gan.pt'))
model.to(device)
model.eval()


def to_torch(img):
    return torch.unsqueeze(torch.tensor(img), dim=0) / 255.

def from_torch(img):
    return (torch.squeeze(img).cpu().detach().numpy() * 255).astype(np.uint8)


def sobel(img):
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    return cv2.bitwise_or(opImgx, opImgy)

def to_sketch(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    inv = 255 - img
    edge = cv2.addWeighted(sobel(img), 0.75, sobel(inv), 0.75, 0)
    return 255 - edge


def normalize(img):
    return (img - immean) / imstd


def simplify(sketch):
    return model(sketch)


data_dir = '../data/CelebAText-HQ'
ids, paths = [], []

with open(f'{data_dir}/datapoints.txt') as fin:
    for id in fin.readlines():
        id = id.strip('\n')
        ids.append(id)
        paths.append(f'{data_dir}/images/{id}.jpg')

print("Done reading data.")


batch_len = 8

for i in range(0, len(ids), batch_len):
    imgs = []
    for path in paths[i : i + batch_len]:
        imgs.append(cv2.imread(path))

    # convert images to sketches and normalize
    sketches = list(map(to_sketch, imgs))

    # simplify sketches
    sketches = torch.stack(list(map(to_torch, sketches)))
    sketches = sketches.to(device).float()
    sketches = simplify(normalize(sketches))

    # convert back to opencv writeable format
    sketches = list(map(from_torch, sketches))

    for j, sketch in enumerate(sketches):
        cv2.imwrite(f'{data_dir}/sketches/{ids[i + j]}.jpg', sketch)
