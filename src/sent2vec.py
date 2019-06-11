# -*- coding: utf-8 -*-
"""sent2vec.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fpjd6LMBeVrj1Rt-yBrb8neoRtWOLcod
"""

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

def get_tagged_sentences(train_csv_corpus,val_csv_corpus):
  
  """Take the train and validation corpus csv files and return the tagged sentences ready to be fed in sent2vec model """
  
  
  train_corpus=train_csv_corpus.drop(['storyid','storytitle'],axis=1)
  val_corpus=val_csv_corpus.drop(['InputStoryid','AnswerRightEnding'],axis=1)
  texts=list(train_corpus.stack())+list(val_corpus.stack())
  tag_texts=[TaggedDocument(simple_preprocess(sentence),[i]) for i,sentence in enumerate(texts)]
  
  
  return tag_texts

def get_sent2vec_model(tag_sentences,load_path=None,refit_model=True):
  
  model = Doc2Vec(vector_size=500, window=8, min_count=5, workers=4,max_vocab_size=30000,epochs=100)
  
  
  if refit_model:
    model.build_vocab(tag_sentences)
    model.train(tag_sentences, total_examples=model.corpus_count, epochs=model.epochs)
    return model
  
  model = Doc2Vec.load(load_path)
  return model

def get_sent2vec(tag_sentences,model,len_train=88161):
  
  """Take the tagged sentences and the sent2vec_model. Return the vector sentences for train and validation separately"""
  
  len_train=len_train
   
  sent2vec=np.array([model.docvecs[sentence] for sentence in range(len(tag_sentences))])
  sent2vec_train=sent2vec[:len_train*5].reshape(-1,5,model.vector_size)
  sent2vec_val=sent2vec[len_train*5:].reshape(-1,6,model.vector_size)
  
  return sent2vec_train,sent2vec_val

def infer_sent2vec(test_file,model):
  test=test_file.drop(columns=['InputStoryid'])
  test_text=list(test.drop(columns=["AnswerRightEnding"]).stack())
  test_sent2vec=[model.infer_vector(simple_preprocess(text)) for text in test_text]
  test_sent2vec=np.array(test_sent2vec).reshape(-1,6,model.vector_size)
  return test_sent2vec
