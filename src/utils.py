import numpy as np
import tensorflow as tf


def get_final_predictions(probabilities, threshold=1):
    """
    Parameters
    ----------
    probabilities:  Array of shape [?, 2] that contains for each prediction the probability to belong to each class
    threshold:      Whether to use a threshold, in case model makes unbalanced predictions

    Assumes that consecutive entries of the probabilities array correspond to the two possible endings for each story
    Returns
    -------
    An array of size [?] with the predicted classes (0 or 1)
    """
    # predictions based on probabilities!
    my_predictions = []

    probabilities_exp = np.exp(probabilities)

    i = 0
    while i < len(probabilities):
        p_first = probabilities_exp[i]
        p_second = probabilities_exp[i + 1]

        p1 = p_first[0] + p_second[1]
        p2 = p_first[1] + p_second[0]

        if p1 > p2 * threshold:
            my_predictions.append(0)
        else:
            my_predictions.append(1)
        i += 2

    return np.array(my_predictions)


# Adapted from: https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
def load_glove_model(path):
    """
    Parameters
    ----------
    path; File containing the glove model

    Returns
    -------
    dict: key corresponds to tokens and values to the embedding of the corresponding token
    """
    print("Loading glove model")
    f = open(path, 'r')
    model = {}

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

