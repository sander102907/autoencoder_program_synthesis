import gensim.downloader as api
import numpy as np
import torch.nn as nn
import torch


class EmbeddingLoader():
    def __init__(self, embedding_dim, pretrained_embedding_name=None):
        self.embedding_dim = embedding_dim

        if pretrained_embedding_name is not None:
            self.vectors = api.load(pretrained_embedding_name)
            self.embedding_dim = self.vectors.vector_size
        else:
            self.vectors = None


    def load_embedding(self, index2token):
        self.emb_weights = np.random.normal(scale=0.6, size=(len(index2token), self.embedding_dim))

        if self.vectors is not None:
            for index, token in index2token.items():
                embedding = np.zeros(self.embedding_dim)
                for label in token:
                    if label in self.vectors:
                        embedding += self.vectors[label]
                    else:
                        embedding += np.random.normal(scale=0.2, size=self.embedding_dim)

                # Get average of the embeddings of the token "phrase"
                embedding /= len(token)

                self.emb_weights[index] = embedding


    def get_embedding_layer(self, freeze=False):
        emb_layer = nn.Embedding.from_pretrained(torch.tensor(self.emb_weights, dtype=torch.float))

        if freeze:
            emb_layer.weight.requires_grad = False
        else:
            emb_layer.weight.requires_grad = True

        return emb_layer