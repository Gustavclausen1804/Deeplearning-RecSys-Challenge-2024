import polars as pl
import torch
import numpy as np

def load_word2vec_embeddings(parquet_path, vocab_size=10000):
    """Load word2vec embeddings from parquet file"""
    # Read parquet file
    df = pl.read_parquet(parquet_path)
    
    # Get embeddings and vocab
    embeddings = df.select('vector').to_numpy()
    vocab = df.select('token').to_numpy()
    
    # Add padding token at index 0
    pad_vector = np.zeros((1, embeddings.shape[1]))
    embeddings = np.vstack([pad_vector, embeddings[:vocab_size-1]])
    
    return torch.FloatTensor(embeddings)
