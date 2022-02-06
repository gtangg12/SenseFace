import spacy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import device
from model import latent_avg
from model import LanguageFeatureModulation, lang_encoder_group_width, lang_encoder_dim, styles_cnt
from model import FaceGenerator, face_loss


spacy_pipeline = spacy.load('en_core_web_md')

def encode_texts(text_list):
    """ Returns list of (sequence len padded to 216, 512) tensor of spacy word embeddings
        for all text in list
    """
    parsed = spacy_pipeline.pipe(text_list,
                                 batch_size=1024,
                                 disable=['tok2vec', 'parser', 'tagger', 'ner', 'attribute_ruler', 'lemmatizer'])
    encoded = []
    for i, doc in tqdm(enumerate(parsed)):
        embeddings = np.array([token.vector for token in doc])
        # 300 is spacy's word embedding dim
        pad = ((0, lang_encoder_group_width * styles_cnt - embeddings.shape[0]), (0, lang_encoder_dim - 300))
        embeddings = torch.from_numpy(np.pad(embeddings, pad))
        encoded.append(embeddings)
    return torch.stack(encoded)


def read_caption(path):
    with open(path) as fin:
        text = fin.read()
    return ' '.join(text.strip('\n').split('\n'))


class LFMDataset():
    def __init__(self, data_dir):
        self.ref_codes = torch.load(f'{data_dir}/img_codes.pt')
        self.inp_codes = torch.load(f'{data_dir}/sketch_codes.pt')
        self.ref_codes -= latent_avg.cpu()
        self.inp_codes -= latent_avg.cpu()

        with open(f'{data_dir}/datapoints.txt') as fin:
            ids = [id.strip('\n') for id in fin.readlines()]
        captions = [read_caption(f'{data_dir}/captions/{id}.txt') for id in ids]
        self.word_embeddings = encode_texts(captions)

    def __len__(self):
        return self.ref_codes.shape[0]

    def __getitem__(self, idx):
        return self.word_embeddings[idx].to(device), \
               self.inp_codes[idx].to(device), \
               self.ref_codes[idx].to(device)


def train(name,
          model,
          train_loader,
          validation_loader,
          criterion,
          optimizer,
          scheduler,
          epochs=200,
          checkpoint_interval=5):

    train_losses, validation_losses = [], []

    for epoch in range(epochs):
        print(f'Epoch-{epoch}')

        train_loss, validation_loss = 0, 0

        model.train()
        for i, (x1, x2, y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            pred = model(x1, x2)
            loss = criterion(pred, y)
            loss.mean().backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for i, (x1, x2, y) in enumerate(tqdm(validation_loader)):
                pred = model(x1, x2)
                loss = criterion(pred, y)
                validation_loss += loss.item()
                if i == 2:
                    print("KAJGSFK")
                    print(pred[3][4][10:20])
                    print(y[3][4][10:20])
                    print()
                    print(pred[5][8][90:100])
                    print(y[5][8][90:100])
                    print()
                    print(pred[7][17][500:510])
                    print(y[7][17][500:510])
                    print()
                    print(pred[7][17][200:210])
                    print(y[7][17][200:210])
                    print()
                    print(pred[1][15][0:10])
                    print(y[1][15][0:10])
                    print()


        scheduler.step(validation_loss)

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        validation_loss /= len(validation_loader)
        validation_losses.append(validation_loss)

        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {validation_loss}')
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model, f'checkpoints/{name}_{(epoch + 1):03d}.pt')

    return train_losses, validation_losses


def main():
    data_dir = 'data/CelebAText-HQ'

    dataset = LFMDataset(data_dir)

    num_train = int(0.85 * len(dataset))
    num_validation = len(dataset) - num_train

    train_dataset, validation_dataset = \
        torch.utils.data.random_split(dataset, [num_train, num_validation])

    train_loader, validation_loader = \
        DataLoader(train_dataset, batch_size=32, shuffle=True), \
        DataLoader(validation_dataset, batch_size=32, shuffle=True)


    model = LanguageFeatureModulation(out_d = 512)
    #model = torch.load('checkpoints/classifier_015.pt')
    model = nn.DataParallel(model, device_ids=[0])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_func = nn.MSELoss()

    losses = train('lfm',
                   model,
                   train_loader,
                   validation_loader,
                   loss_func,
                   optimizer,
                   scheduler)

    with open('losses.txt') as fout:
        for i in range(len(losses[0])):
            fout.write(f'{losses[0][i]}, {losses[1][i]}\n')

    '''
    model = FaceGenerator()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_func = face_loss

    losses = train(model,
                   train_loader,
                   validation_loader,
                   loss_func,
                   optimizer,
                   scheduler)

    with open('losses_finetune.txt') as fout:
        for i in range(len(losses[0])):
            fout.write(f'{losses[0][i]}, {losses[1][i]}\n')
    '''


if __name__ == '__main__':
    main()
