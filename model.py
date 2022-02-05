import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FuseBlock(nn.Module):
    def __init__(self, num_layers, in_d, out_d):
        super(Fuse, self).__init__()
        modules = []
        modules += [nn.Linear(in_d, out_d)]
        for i in range(num_layers):
            modules += [
                nn.BatchNorm1d(out_d),
                nn.ReLU(),
                nn.Linear(out_d, out_d)
            ]
        self.fuse = nn.Sequential(*modules)
        self.affine = nn.Linear(out_d, out_d)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        x = self.fuse(x)
        x = self.affine(x)
        return x


class LanguageFeatureModulation(nn.Module):
    def __init__(self, out_d, opts):
        super(LangFM, self).__init__()
        self.opts = opts
        self.styles_cnt = self.opts['n_styles']

        # WARNING: torch 1.7.1 has batch_first by default unlike later versions
        lang_encoder_layer = \
            nn.TransformerEncoderLayer(d_model=512, nhead=4)
        self.lang_encoder = \
            nn.TransformerEncoder(lang_encoder_layer, num_layers=6)

        modules = []
        for i in range(self.styles_cnt):
            modules += [FuseBlock(opts['lang_dim'] + out_d, out_d)]
        self.style_layers = nn.Sequential(*modules)

    def forward(self, codes, word_embeddings):
        lang_hidden = self.lang_encoder(word_embeddings)
        lang_hidden = lang_hidden.view(-1, self.styles_cnt, 12, 512)
        lang_codes = self.norm_latent(torch.sum(lang_hidden, dim=2, keepdim=True))

        latents = []
        for i in range(self.styles_cnt):
            latents.append(self.style_layers[i](
                lang_codes[:, i, ...], torch.unsqueeze(codes[:, i, :], dim=1)
            ))
        return torch.stack(latents, dim=1)

    def norm_latent(self, codes):
        sum = torch.sum(codes, dim=-1) / 512.
        std = torch.std(codes, dim=-1)
        codes = codes - torch.unsqueeze(sum, dim=-1)
        codes = codes / torch.unsqueeze(std, dim=-1)
        return codes
