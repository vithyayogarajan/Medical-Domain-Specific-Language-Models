import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf

#keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import regularizers
from tensorflow import concat
from tensorflow.keras import layers
import time
import codecs

import re
import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def adaptive_gpu_memory():
    tf_version = int(tf.__version__.split('.')[0])
    if tf_version > 1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        from tensorflow.keras.backend import set_session
        set_session(sess)
        return tf_version


adaptive_gpu_memory()


df = pd.read_csv("fungal30814_73label_halfdissummary.csv")

labels = ['CAT_1','CAT_2','CAT_3','CAT_4','CAT_5','CAT_8','CAT_9','CAT_10','CAT_11','CAT_12','CAT_13','CAT_15','CAT_16','CAT_20','CAT_21','CAT_22','CAT_23','CAT_24','CAT_27','CAT_30','CAT_31','CAT_33','CAT_34','CAT_35','CAT_38','CAT_39','CAT_40','CAT_41','CAT_110','CAT_111','CAT_112','CAT_117','CAT_320','CAT_322','CAT_324','CAT_325','CAT_420','CAT_421','CAT_451','CAT_461','CAT_462','CAT_463','CAT_464','CAT_465','CAT_481','CAT_482','CAT_485','CAT_486','CAT_491','CAT_494','CAT_510','CAT_513','CAT_540','CAT_541','CAT_542','CAT_556','CAT_562','CAT_567','CAT_569','CAT_572','CAT_575','CAT_590','CAT_599','CAT_601','CAT_614','CAT_616','CAT_681','CAT_682','CAT_683','CAT_686','CAT_730','CAT_790','CAT_999']


train, test = train_test_split(df, test_size=0.2)


#load embeddings
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('T100SG.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))



MAX_NB_WORDS = 100000
max_seq_len = 2000
num_classes = 73


raw_docs_train1 = train['text1'].tolist()
raw_docs_test1 = test['text1'].tolist()


print("tokenizing input data 1...")
tokenizer1 = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer1.fit_on_texts(raw_docs_train1 + raw_docs_test1) 
word_seq_train1 = tokenizer1.texts_to_sequences(raw_docs_train1)
word_seq_test1 = tokenizer1.texts_to_sequences(raw_docs_test1)
word_index1 = tokenizer1.word_index
print("dictionary size: ", len(word_index1))

#pad sequences
word_seq_train1 = pad_sequences(word_seq_train1, maxlen=max_seq_len)
word_seq_test1 = pad_sequences(word_seq_test1, maxlen=max_seq_len)


raw_docs_train2 = train['text2'].tolist()
raw_docs_test2 = test['text2'].tolist()


print("tokenizing input data 2 ...")
tokenizer2 = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer2.fit_on_texts(raw_docs_train2 + raw_docs_test2) 
word_seq_train2 = tokenizer2.texts_to_sequences(raw_docs_train2)
word_seq_test2 = tokenizer2.texts_to_sequences(raw_docs_test2)
word_index2 = tokenizer2.word_index
print("dictionary size: ", len(word_index2))

#pad sequences
word_seq_train2 = pad_sequences(word_seq_train2, maxlen=max_seq_len)
word_seq_test2 = pad_sequences(word_seq_test2, maxlen=max_seq_len)


embed_dim = 100 

#embedding matrix
print('preparing embedding matrix 1...')
words_not_found1 = []
nb_words1 = min(MAX_NB_WORDS, len(word_index1))
embedding_matrix1 = np.zeros((nb_words1, embed_dim))
for word, i in word_index1.items():
    if i >= nb_words1:
        continue
    embedding_vector1 = embeddings_index.get(word)
    if (embedding_vector1 is not None) and len(embedding_vector1) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix1[i] = embedding_vector1
    else:
        words_not_found1.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix1, axis=1) == 0))

#embedding matrix
print('preparing embedding matrix 2...')
words_not_found2 = []
nb_words2 = min(MAX_NB_WORDS, len(word_index2))
embedding_matrix2 = np.zeros((nb_words2, embed_dim))
for word, i in word_index2.items():
    if i >= nb_words2:
        continue
    embedding_vector2 = embeddings_index.get(word)
    if (embedding_vector2 is not None) and len(embedding_vector2) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix2[i] = embedding_vector2
    else:
        words_not_found2.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix2, axis=1) == 0))


num_filters = 100
filter_sizes=[2,3,4,5]
training_dropout_keep_prob=0.5

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model_input1 = Input(shape=(max_seq_len, ), name="text1")
model_input2 = Input(shape=(max_seq_len, ), name="text2")
  
z1 =  Embedding(nb_words1 + 1, 
                  embed_dim, 
                   input_length=max_seq_len, embeddings_regularizer=regularizers.l2(0.003),
                   name="embedding1")(model_input1)
z2 =  Embedding(nb_words2 + 1, 
                  embed_dim, 
                   input_length=max_seq_len, embeddings_regularizer=regularizers.l2(0.003),
                   name="embedding2")(model_input2)    
 
    # Convolutional block
conv_blocks1 = []
for sz in filter_sizes:
    conv1 = Convolution1D(filters=num_filters,                         
                     kernel_size=sz,
                     padding="valid",
                     activation="relu",
                     strides=1)(z1)
    window_pool_size =  max_seq_len  - sz + 1 
    conv1 = MaxPooling1D(pool_size=window_pool_size)(conv1)  
    conv1 = Flatten()(conv1)
    conv_blocks1.append(conv1)

#concatenate
z1 = Concatenate()(conv_blocks1) if len(conv_blocks1) > 1 else conv_blocks1[0]

    # Convolutional block
conv_blocks2 = []
for sz in filter_sizes:
    conv2 = Convolution1D(filters=num_filters,                         
                     kernel_size=sz,
                     padding="valid",
                     activation="relu",
                     strides=1)(z2)
    window_pool_size =  max_seq_len  - sz + 1 
    conv2 = MaxPooling1D(pool_size=window_pool_size)(conv2)  
    conv2 = Flatten()(conv2)
    conv_blocks2.append(conv2)

#concatenate
z2 = Concatenate()(conv_blocks2) if len(conv_blocks2) > 1 else conv_blocks2[0]

z = layers.concatenate([z1,z2])
z = Dropout(training_dropout_keep_prob)(z)

#score prediction
model_output = Dense(num_classes, activation="sigmoid", name="modeloutput")(z)

    #creating model
model = Model(
     inputs = [model_input1,model_input2],
     outputs = [model_output],
    )
model.compile(loss="binary_crossentropy", optimizer="adam",  metrics=['acc'])
      
print(model.summary())

  

history=model.fit(
    {"text1":word_seq_train1,"text2":word_seq_train2},train[labels], callbacks=callback, 
                  batch_size=32, epochs=600, 
                  validation_split=0.1, 
                  verbose=2)



y_out = model.predict(
    {"text1": word_seq_test1,"text2": word_seq_test2},
    batch_size=32,
)
y_pred = np.where(y_out > 0.5, 1, 0)


print(classification_report(test[labels], y_pred,digits=4))

with open('output_dualnewCNN_fungal.txt', 'w') as f:
    print('classification report', classification_report(test[labels], y_pred,digits=4),file=f)






