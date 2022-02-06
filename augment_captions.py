import os
import tensorflow as tf
from deepface import DeepFace
from train import read_caption, encode_texts
from precompute_latent_codes import read_ids, read_filepaths

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_dir = 'data/CelebAText-HQ'

ids = read_ids()

img_paths = read_filepaths('images', 'jpg')

batch_len = 1024
for j in range(0, len(ids), batch_len):
    img_results = DeepFace.analyze(img_paths[j : j + batch_len],
                                   actions = ['age', 'gender', 'race', 'emotion'],
                                   enforce_detection=False)
    for i, id in enumerate(ids[j : j + batch_len]):
        results = img_results[f'instance_{i + 1}']
        age = results['age']
        gender = results['gender']
        race = results['dominant_race']
        emotion = results['dominant_emotion']

        caption_path = f'{data_dir}/captions/{id}.txt'
        caption = read_caption(caption_path)
        caption = f'{age}. {gender}. {race}. {emotion}. {caption}'

        with open(caption_path, 'w') as fout:
            fout.write(caption)
