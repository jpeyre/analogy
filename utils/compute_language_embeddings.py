"""
Compute language embeddings for the labels
NB: we release our language embeddings so you won't need to run this script. We only provide it as indication and in case you wish to compute word embedding for new words. 
We use a Word2vec model trained on GoogleNews: 
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
"""
# Get language embedding : in dev -> to put in data loader
from __future__ import division
from gensim.models import KeyedVectors
import pickle
import os.path as osp
import numpy as np

# Load model
path_to_word2vec = './data/GoogleNews-vectors-negative300.bin'
word_vectors = KeyedVectors.load_word2vec_format(path_to_word2vec, binary=True)  # C binary format

# Example for HICO-DET
data_path = '/sequoia/data2/jpeyre/iccv19_final/datasets/hico'

vocab = pickle.load(open(osp.join(data_path, 'vocab.pkl'),'rb'))
print('Found %d words in vocab' %len(vocab))

# Manual mapping of some of COCO objects
vocab[vocab.index('teddy bear')] = 'teddybear'
vocab[vocab.index('fire hydrant')] = 'fire_hydrant'
vocab[vocab.index('tennis racket')] = 'tennis_racket'
vocab[vocab.index('hot dog')] = 'hotdog'
vocab[vocab.index('cell phone')] = 'cellphone'
vocab[vocab.index('wine glass')] = 'wineglass'

# For the rest do average and map 'no_interaction' and 'background' directly to random
embeddings = np.zeros((len(vocab),300))
for j in range(len(vocab)):

    word = vocab[j]
    if word in word_vectors.vocab:
       embeddings[j,:] = word_vectors[word] 

    # Map 'no_interaction' and 'background' directly to random (xavier init)
    elif word in ['no interaction', 'background']:
        embeddings[j,:] = np.random.normal(0, scale=1/np.sqrt(300), size=300) 

    # Usually a compound word -> either separated by _ or space. Take average embedding (alternative strategy : focus on first word for verb/ second word for noun)
    elif word not in word_vectors.vocab:
        words = word.split('_') if '_' in word else word.split(' ')
        embeddings[j,:] = (word_vectors[words[0]] + word_vectors[words[1]])/2
         
    else:
        print('Could not find embedding for word %s' %word)


save_file = osp.join(data_path, 'pretrained_embeddings_w2v.pkl')
if osp.exists(save_file):
    answer = raw_input("File %s already exists. Continue: yes/no?" %save_file)
    assert answer=='yes', 'Please speficy another file name'


pickle.dump(embeddings, open(osp.join(data_path, 'pretrained_embeddings_w2v.pkl'), 'wb'), protocol=2)

