import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

"""
    Helpers
"""
def freeze_model_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def get_keys(ckpt, name):
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    return {k[len(name) + 1:]: v for k, v in ckpt.items() if k[:len(name)] == name}


def load_ckpt(path):
    ckpt = torch.load(path)
    ckpt['opts']['output_size'] = 1024
    ckpt['opts']['n_styles'] = int(math.log(ckpt['opts']['output_size'], 2)) * 2 - 2

    model = GradualStyleEncoder(50, 'ir_se', ckpt['opts'])
    model.load_state_dict(get_keys(ckpt, 'encoder'))
    model.to(device)
    model.eval()

    return model


"""
    StyleGAN2
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
    LFM
"""
class FuseBlock(nn.Module):
    def __init__(self, num_layers, in_d, out_d):
        super(FuseBlock, self).__init__()
        modules = []
        modules += [nn.Linear(in_d, in_d), nn.ReLU(), nn.Linear(in_d, out_d)]
        for i in range((num_layers - 1) // 2):
            modules += [
                nn.BatchNorm1d(out_d),
                nn.ReLU(),
                nn.Linear(out_d, out_d),
                nn.BatchNorm1d(out_d),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(out_d, out_d),
            ]
        self.fuse = nn.Sequential(*modules)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        x = self.fuse(x)
        return x


lang_encoder_group_width = 12
lang_encoder_dim = 512
styles_cnt = 18

class LanguageFeatureModulation(nn.Module):
    def __init__(self, out_d):
        super(LanguageFeatureModulation, self).__init__()

        # WARNING: torch 1.7.1 has batch_first by default unlike later versions
        lang_encoder_layer = \
            nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.lang_encoder = \
            nn.TransformerEncoder(lang_encoder_layer, num_layers=8)

        '''
        style_encoder_layer = \
            nn.TransformerEncoderLayer(d_model=512, nhead=4)
        self.style_encoder = \
            nn.TransformerEncoder(style_encoder_layer, num_layers=2)
        '''
        modules = []
        for i in range(styles_cnt):
            modules += [FuseBlock(4, lang_encoder_dim + out_d, out_d)]
        self.styles = nn.Sequential(*modules)

        self.affine = nn.ModuleList()
        for i in range(styles_cnt):
            module = nn.Sequential(nn.Linear(out_d, out_d),
                                   nn.BatchNorm1d(out_d),
                                   nn.Linear(out_d, out_d),
                                   nn.BatchNorm1d(out_d),
                                   nn.Dropout(),
                                   nn.Linear(out_d, out_d))
            self.affine.append(module)


    def forward(self, word_embeddings, codes):
        lang_hidden = self.lang_encoder(word_embeddings)
        lang_hidden = lang_hidden.view(-1, styles_cnt, lang_encoder_group_width, 512)
        lang_codes = self.norm_latent(torch.sum(lang_hidden, dim=2))

        latents = []
        for i in range(styles_cnt):
            style = self.styles[i](lang_codes[:, i, :], codes[:, i, :])
            style = self.affine[i](style)
            latents.append(style)
        latents = torch.stack(latents, dim=1)
        return latents

    def norm_latent(self, codes):
        sum = torch.sum(codes, dim=-1) / 512.
        std = torch.std(codes, dim=-1)
        codes = codes - torch.unsqueeze(sum, dim=-1)
        codes = codes / torch.unsqueeze(std, dim=-1)
        return codes

    def prepare_fine_tune(self):
        freeze_model_grad(self)
        for param in self.affine.parameters():
            param.requires_grad = True


"""
    Actual Inference Model
"""
class FaceGenerator:
    def __init__(self):
        super(FaceGenerator, self).__init__()
        self.encoder = load_ckpt('pretrained_models/psp_celebs_sketch_to_face.pt')
        self.decoder = nn.Sequential(stylegan, face_pool)
        self.lang_fm = LanguageFeatureModulation(out_d = 512)

    def forward(self, word_embeddings, codes):
        assert self.fine_tune, 'Model not in fine tune mode'
        return run(word_embeddings, codes)

    def infer(self, word_embeddings, sketch):
        codes = self.encoder(sketch)
        return run(word_embeddings, codes)

    def run(word_embeddings, codes):
        styles = self.lang_fm(word_embeddings, codes)
        styles = styles + latent_avg
        images, _ = self.decoder([styles])
        return images, styles

    def prepare_fine_tune(self):
        self.fine_tune = true
        self.lang_fm.prepare_fine_tune()
        freeze_model_grad(self.encoder)
        freeze_model_grad(self.decoder)


"""
    Loss function for fine-tuning FaceGenerator once LFM has been trained
"""
from lpips import LPIPS
from arcface.backbone import Backbone

perceptual_similarity = LPIPS(net='alex').to(device)
l2 = nn.MSELoss().to(device)

facenet = Backbone(input_size=112, mode='ir_se')
facenet.load_state_dict(torch.load('pretrained_models/model_ir_se50.pt'))
facenet.to(device)
facenet.eval()

def extract_feat(img):
    """ Requires input image dim to be (3, 256, 256) """
    return facenet(face_pool(img[:, :, 35:223, 32:220]))

def face_similarity(img1, img2):
    return 1.0 - F.cosine_similarity(extract_feat(img1), extract_feat(img2))


def face_loss(output, target, latent):
    latent = latent - latent_avg
    loss1 = l2(output, target)
    loss2 = torch.sum(perceptual_similarity(output, target)) / latent.shape[0]
    loss3 = torch.sum(face_similarity(output, target)) / latent.shape[0]
    loss4 = torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]

    loss = 0.010 * loss1 + 0.800 * loss2 + 1.000 * loss3 + 0.005 * loss4

    return loss, torch.tensor([loss1.item(), loss2.item(), loss3.item(), loss4.item()])
