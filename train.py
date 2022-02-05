import spacy


spacy_pipeline = spacy.load('en_core_web_md')

def encode_texts(text_list):
    """ Returns list of (sequence len, 512) tensor of spacy word embeddings
        for all text in list
    """
    parsed = spacy_pipeline.pipe(text_list,
                                 batch_size=1024,
                                 disable=['parser', 'tagger', 'ner', 'attribute_ruler', 'lemmatizer'])
    encoded = []
    for i, doc in tqdm(enumerate(parsed)):
        #if i == 1024:
        #    break
        embeddings = np.array([token.vector for token in doc])
        pad = ((0, 12 * 18 - embeddings.shape[0]), (0, 212))
        embeddings = torch.from_numpy(np.pad(embeddings, pad))
        encoded.append(embeddings)
    return encoded


#DONT FORGET TO SUB LATENT AVG FROM CODES!!!!!
