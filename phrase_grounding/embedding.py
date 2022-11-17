from tqdm import tqdm
import numpy as np
import fasttext

class Glove():
    def __init__(self, path):
        self.glove = self.load_glove(path)
        self.num_vocab = len(self.glove)
        self.emb_dim = self.glove[list(self.glove.keys())[0]].shape[0]

    def load_glove(self, path):
        glove = {}
        with open(path, 'r') as f:
            for line in tqdm(f):
                vals = line.rstrip().split(' ')
                glove[vals[0]] = np.array([float(x) for x in vals[1:]])
        
        return glove

    def get_vector(self, word):
        tokens = word.lower().split(' ')
        invocab = 0
        vector = np.zeros(self.emb_dim) 
        for token in tokens:
            if token in self.glove:
                vector += self.glove[token]
                invocab += 1
        
        return vector / invocab if invocab != 0 else vector

    def distance(self, x, y, similarity='cosine'):
        x_vec = self.get_vector(x)
        y_vec = self.get_vector(y)

        if similarity == 'cosine':
            if np.linalg.norm(x_vec) * np.linalg.norm(y_vec) != 0:
                return np.dot(x_vec, y_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(y_vec))
            else:
                return np.linalg.norm(x_vec - y_vec)
        elif similarity == 'norm':
            return np.linalg.norm(x_vec - y_vec)


class FastText():
    def __init__(self, path):
        self.ft = fasttext.load_model(path)
        self.num_vocab = len(self.ft.words)
        self.emb_dim = self.ft.get_dimension()

    def get_vector(self, word):
        tokens = word.lower().split(' ')
        invocab = 0
        vector = np.zeros(self.emb_dim) 
        for token in tokens:
            if token in self.ft:
                vector += self.ft[token]
                invocab += 1
        
        return vector / invocab if invocab != 0 else vector

    def distance(self, x, y, similarity='cosine'):
        x_vec = self.get_vector(x)
        y_vec = self.get_vector(y)

        if similarity == 'cosine':
            if np.linalg.norm(x_vec) * np.linalg.norm(y_vec) != 0:
                return np.dot(x_vec, y_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(y_vec))
            else:
                return np.linalg.norm(x_vec - y_vec)
        elif similarity == 'norm':
            return np.linalg.norm(x_vec - y_vec)