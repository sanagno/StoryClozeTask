import sys
import numpy as np
import pandas as pd

# Custom dependencies 
import data
from language_model_v3 import LanguageModel

if __name__ == '__main__':

    # Load data
    dataloader = data.fetch_data()

    train_stories = dataloader['train']
    valid_stories, valid_labels = dataloader['valid']

    # Construct the vocabulary
    vocab, inverse_vocab, max_len = data.construct_vocab(train_stories)

    encoded_train_context_, _ = data.encode_text(train_stories, max_len, vocab)

    # Append max_len tokens to the training context (for consistency during training)
    train_pads = np.full(shape=(encoded_train_context_.shape[0], max_len), fill_value=vocab['<pad>'], dtype=int)

    encoded_train_context = np.hstack((encoded_train_context_, train_pads))

    # Encode validation data for finetuning (context + wrong ending)
    val_finetune_data = data.encode_valid_text_for_fine_tunning(valid_stories, valid_labels, max_len, vocab)

    # Train the language model 
    language_model = LanguageModel(vocab, inverse_vocab, max_len)
    language_model.train(encoded_train_context, val_finetune_data)

    # Encode training stories and correct endings for condintional generation
    context_nopads, endings_nopad = data.encode_train_text_for_conditional_generation(train_stories, max_len, vocab)

    # Generate negative endings 
    generated_endings = language_model.ending_generation(context_nopads, endings_nopad)

    # Save generated ending (in the same order as training stories)
    np.save('incorrect_endings_language_model.npy', generated_endings)

