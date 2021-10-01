# ## Hierarchical LSTM or GRU Attention
# This model was implemented based on Yang et al. (2016) Hierarchical Attention networks for document classification.
 
# https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf

import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from tensorflow.keras.layers import Concatenate, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report
from tensorflow.keras.layers.core import *
from tensorflow.keras.layers import merge, dot, add
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import re
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from nltk import tokenize
import re
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import nltk


#reading file
df = pd.read_csv('18label_mimic_dis.csv') ### pre-processed text with 18 labels

note_sentences = []
notes = []

for idx in range(df.shape[0]):
    # for every note
    text = df["TEXT"][idx]
    notes.append(text)
    sentences = tokenize.sent_tokenize(text)
    note_sentences.append(sentences)   

note_sentences_length =[len(x) for x in note_sentences]
print("Average number of sentences in a note: ", np.mean(note_sentences_length))  
print("Max number of sentences in a note: ", max(note_sentences_length))

MAX_NB_WORDS = None
MAX_VOCAB = None # to limit original number of words (None if no limit)
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(notes)


notes_tokenized = []
sentence_length= []
for one_note_sentences in note_sentences:
    note_sentences_tokenized = []
    for sentence in one_note_sentences:
        sentence_words = text_to_word_sequence(sentence)
        sentence_length.append(len(sentence_words))
        note_sentences_tokenized.append(sentence_words)
    notes_tokenized.append(note_sentences_tokenized)
print(len(notes_tokenized))                                        


print("Average number of words in a sentence: ", np.mean(sentence_length))  
print("Max number of words in a sentence: ", max(sentence_length))


dictionary = tokenizer.word_index

MAX_SENTS = 150
MAX_SENT_LENGTH  = 160

MAX_NB_WORDS = len(tokenizer.word_index)  #vocabulary length
note_matrix = np.zeros((len(notes), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, one_note_sentences in enumerate(note_sentences):
    for j, sentence in enumerate(one_note_sentences):
        if j< MAX_SENTS:
            wordTokens = text_to_word_sequence(sentence)
            k=0
            for _, word in enumerate(wordTokens):
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                    note_matrix[i,j,k] = tokenizer.word_index[word]
                    k+=1


# Creates an embedding Matrix
# Based on https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

def embedding_matrix(f_name, dictionary, EMBEDDING_DIM, verbose = True, sigma = None):
    """Takes a pre-trained embedding and adapts it to the dictionary at hand
        Words not found will be all-zeros in the matrix"""

    # Dictionary of words from the pre trained embedding
    pretrained_dict = {}
    with open(f_name, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            pretrained_dict[word] = coefs

    # Default values for absent words
    if sigma:
        pretrained_matrix = sigma * np.random.rand(len(dictionary) + 1, EMBEDDING_DIM)
    else:
        pretrained_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
    
    # Substitution of default values by pretrained values when applicable
    for word, i in dictionary.items():
        vector = pretrained_dict.get(word)
        if vector is not None:
            pretrained_matrix[i] = vector

    if verbose:
        print('Vocabulary in notes:', len(dictionary))
        print('Vocabulary in original embedding:', len(pretrained_dict))
        inter = list( set(dictionary.keys()) & set(pretrained_dict.keys()) )
        print('Vocabulary intersection:', len(inter))

    return pretrained_matrix, pretrained_dict

def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=101):
    """Splits the input and labels into 3 sets"""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size+test_size), random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(val_size+test_size), random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


EMBEDDING_DIM = 50 
EMBEDDING_MATRIX= []
EMBEDDING_LOC = 'Emd50.txt' # pretrained 50 dimensional embeddings
EMBEDDING_MATRIX, embedding_dict = embedding_matrix(EMBEDDING_LOC,
                                                                  dictionary, EMBEDDING_DIM, verbose = True, sigma=True)
labels = ['blood', 'circulatory', 'congenital', 'digestive', 'endocrine',
       'genitourinary', 'infectious', 'injury', 'mental', 'muscular',
       'neoplasms', 'nervous', 'pregnancy', 'prenatal', 'respiratory', 'skin',
       'symptoms', 'E and V']

Y = df[labels]

#split sets
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    note_matrix, Y, val_size=0.2, test_size=0.1, random_state=101)
print("Train: ", X_train.shape, y_train.shape)
print("Validation: ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)


# ## Hierarchical Attention NN
## Multi-Label Classification of Patient Notes:Case Study on ICD Code Assignment
## https://github.com/talbaumel/MIMIC
## https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py


def attention_layer(inputs, TIME_STEPS,lstm_units, i='1'):

    inputs= Dropout(0.5)(inputs)
    u_it = TimeDistributed(Dense(lstm_units, activation='tanh',
                                 kernel_regularizer=regularizers.l2(0.0001),
                                 name='u_it'+i))(inputs)

    u_it= Dropout(0.5)(u_it)
    att = TimeDistributed(Dense(1, 
                                kernel_regularizer=regularizers.l2(0.0001),
                                bias=False))(u_it)                         
    att = Reshape((TIME_STEPS,))(att)                                                       
    att = Activation('softmax', name='alpha_it_softmax'+i)(att) 

    
    s_i =dot([att, inputs],axes=1, normalize=False, name='s_i_dot'+i) 
  
    
    return s_i


def hierarhical_att_model(MAX_SENTS, MAX_SENT_LENGTH, embedding_matrix,
                         max_vocab, embedding_dim, 
                         num_classes,training_dropout):
   
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    
    embedded_sequences = Embedding(max_vocab + 1,
                           embedding_dim,
                           weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH, embeddings_regularizer=regularizers.l2(0.0001),
                            trainable=True)(sentence_input)
    
    gru_dim = 50
    h_it_sentence_vector = Bidirectional(GRU(gru_dim, return_sequences=True))(embedded_sequences)
    #h_it_sentence_vector =  Bidirectional(LSTM(gru_dim, return_sequences=True))(embedded_sequences)

    words_attention_vector = attention_layer(h_it_sentence_vector,MAX_SENT_LENGTH,gru_dim) 

    sentEncoder = Model(sentence_input, words_attention_vector)
    
    print(sentEncoder.summary())

    # SENTENCE LAYER
    
    note_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    note_encoder = TimeDistributed(sentEncoder)(note_input)
    document_vector = Bidirectional(GRU(gru_dim, return_sequences=True))(note_encoder)
    #document_vector = Bidirectional(LSTM(gru_dim, return_sequences=True))(note_encoder)
    
	#attention layer
    sentences_attention_vector = attention_layer(document_vector,MAX_SENTS,gru_dim) 
    
	# output layer
    z = Dropout(training_dropout)(sentences_attention_vector)
    preds = Dense(num_classes, activation='sigmoid', name='preds')(z)
    
    #model
    model = Model(note_input, preds)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
   
    print("Hierachical Attention GRU")
    print(model.summary())

    return model


model = hierarhical_att_model(MAX_SENTS, MAX_SENT_LENGTH, 
                         max_vocab=MAX_NB_WORDS, embedding_dim=EMBEDDING_DIM , embedding_matrix=EMBEDDING_MATRIX ,
                         num_classes=18,training_dropout=0.5)


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model.fit(X_train, y_train, batch_size=50, epochs=200, validation_data=(X_val, y_val), verbose=1,callbacks=callback)


y_out = model.predict(X_test, batch_size=100)
y_pred = np.where(y_out > 0.5, 1, 0)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=labels))






