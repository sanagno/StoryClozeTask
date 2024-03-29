import numpy as np 
import tensorflow as tf 

def load_glove_model(path):
    """
    Adapted from: https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python

    Parameters
    ----------
    path : String 
        Path to GloVe model.

    Returns
    -------
    model : dict
        Keys correspond to tokens and values to the embedding of the corresponding token.
    """
    
    f = open(path, 'r')
    
    model = {}
    print("Loading glove model")
    
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding

    print("Done.", len(model), " words loaded!")

    return model

def load_embedding(session, vocab, emb, path, dim_embedding):
    """
    Parameters
    ----------
    session:        Tensorflow session object
    vocab:          A dictionary mapping token strings to vocabulary IDs
    emb             Embedding tensor of shape vocabulary_size x dim_embedding
    path            Path to embedding file
    dim_embedding   Dimensionality of the external embedding.
    -------
    Updates the parameters of the specified weight vector
    """
    vocab_size = len(vocab)

    model = load_glove_model(path)

    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.items():
        if tok in model:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab_size))

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})  # here, embeddings are actually set