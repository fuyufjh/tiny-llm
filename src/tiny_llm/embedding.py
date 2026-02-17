import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight # vocab_size x embedding_dim

    def __call__(self,
                 x: mx.array # N.. (tokens)
                 ) -> mx.array:
        # lookup embedding vector for each token
        return self.weight[x, :] # N.. x embedding_dim
        

    def as_linear(self,
                  x: mx.array # N.. x embedding_dim
                  ) -> mx.array:
        # To project the embedding back to vocab space, we can calculate the dot product (similarity) 
        # between the input and each embedding vector.
        # This is equivalent to a linear layer with the embedding weight as the weight matrix.
        return x @ self.weight.T
