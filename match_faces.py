import numpy as np
#import scann
import torch
from torchvision import transforms
from model import device
from model import load_ckpt
from model import extract_feat, face_similarity
from model import FaceGenerator
from model import stylegan, face_pool, latent_avg
from precompute_latent_codes import imread, imsave, to_torch, from_torch, unnormalize, generate_codes
from train import read_caption, encode_texts


'''
def knn(database, queries, k=5):
    """ Return top 5 nearest neighbors for each embedding query from database
        neighbors: N x k array of top k document indicies, where N is number of queries
    """
    normalized_database = dataset / np.linalg.norm(database, axis=1)[:, np.newaxis]

    searcher = scann.scann_ops_pybind.builder(
            normalized_database, k, "dot_product"
        ).tree(
            num_leaves=2000, num_leaves_to_search=100, training_sample_size=70000
        ).score_ah(
            2, anisotropic_quantization_threshold=0.2
        ).reorder(100).build()

    neighbors, _ = searcher.search_batched(queries)
    return np.array(neighbors)
'''

facegen = FaceGenerator()
#facegen = torch.load('[insert path]')
facegen.to(device)


latent_mask = [1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17]
alpha = 1

def generate_augmentations(codes, word_embeddings, n_outputs_to_generate=16):
    _, codes = facegen(word_embeddings, codes - latent_avg)
    codes = codes + latent_avg

    outputs = []
    to_inject = np.random.randn(n_outputs_to_generate, 512).astype('float32') # standard normal
    for latent in to_inject:
        # go from w latent vector to w+ styles
        latent = torch.from_numpy(latent).unsqueeze(0).to(device)
        _, codes_other = stylegan([latent], input_is_latent=False, return_latents=True)
        # mix styles
        codes_fused = torch.clone(codes)
        for i in latent_mask:
            codes_fused[:, i, :] = codes[:, i, :] + alpha * codes_other[:, i, :]
        # gen images
        codes_fused = codes_fused.detach()
        images, _ = stylegan([codes_fused], input_is_latent=True, randomize_noise=False)
        images = face_pool(images)
        outputs.extend([x for x in images])
    return outputs


demo_name = '585'

if __name__ == '__main__':
    torch.cuda.empty_cache()

    '''
    codes = torch.load('perturbed_443_glasses.pth').detach().to(device).float()
    codes = torch.unsqueeze(codes, dim=0)

    print(codes.shape)

    imgs, _ = stylegan([codes], input_is_latent=True, randomize_noise=False)
    imgs = face_pool(imgs)
    imsave(f'tmp.jpg', from_torch(unnormalize(imgs[0])))
    exit()
    '''

    sketch_encoder = load_ckpt('pretrained_models/psp_celebs_sketch_to_face.pt')
    print("Sketch Encoder Loaded")

    sketch_encoder_transforms = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    sketch_codes = generate_codes([f'demo/{demo_name}_sketch.jpg'],
                                  sketch_encoder,
                                  sketch_encoder_transforms,
                                  grayscale=True).to(device)

    caption = read_caption(f'demo/{demo_name}_caption.txt')
    word_embeddings = encode_texts([caption]).to(device)


    outputs = generate_augmentations(sketch_codes, word_embeddings)
    for i, img in enumerate(outputs):
        imsave(f'demo/{demo_name}_augment/{i}.jpg', from_torch(unnormalize(img)))

    '''
    imgs, _ = facegen(word_embeddings, sketch_codes - latent_avg)
    imsave(f'demo/{demo_name}_synth_text.jpg', from_torch(unnormalize(imgs[0])))
    '''

    '''
    print(sketch_codes.shape)
    imgs, _ = stylegan([sketch_codes], input_is_latent=True, randomize_noise=False)
    imgs = face_pool(imgs)
    imsave(f'demo/{demo_name}_synth_base.jpg', from_torch(unnormalize(imgs[0])))
    '''
