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

import transformers

from transformers import BertTokenizer, BertModel, BertConfig
from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_linear_schedule_with_warmup
)

from transformers import *

import re
import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


df = pd.read_csv("data_dis_ecg_cardio.csv") ### discharge summary and ecg where discharge summary is used for CNNText and ecg for BERT


labels = ['ICD9_25000', 'ICD9_25002', 'ICD9_2689',
       'ICD9_2720', 'ICD9_2721', 'ICD9_2724', 'ICD9_4010', 'ICD9_4011',
       'ICD9_4019', 'ICD9_40290', 'ICD9_40291', 'ICD9_41011', 'ICD9_41091',
       'ICD9_412', 'ICD9_4139', 'ICD9_41400', 'ICD9_41401', 'ICD9_4149',
       'ICD9_42731', 'ICD9_4280', 'ICD9_4292', 'ICD9_43310', 'ICD9_43311',
       'ICD9_43491', 'ICD9_4370', 'ICD9_4371', 'ICD9_79029', 'ICD9_V173',
       'ICD9_V1749', 'ICD9_V5869']


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
max_seq_len = 3000
num_classes = 30

train, test = train_test_split(df, test_size=0.2)
df_train, valid = train_test_split(train, test_size=0.2)

xTrain=df_train['text1'] ## discharge summary
xBTrain = df_train['text2'] ## ecg text
y_train=df_train[labels]

xValid=valid['text1']
xBValid = valid['text2']
y_valid=valid[labels]

xTest=test['text1']
xBTest = test['text2']
y_test=test[labels]


del df, train

raw_docs_train = xTrain.tolist()
raw_docs_test = xTest.tolist()
raw_docs_val = xValid.tolist()

print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(raw_docs_train + raw_docs_test + raw_docs_test) 
word_seq_train = tokenizer.texts_to_sequences(raw_docs_train)
word_seq_test = tokenizer.texts_to_sequences(raw_docs_test)
word_seq_val = tokenizer.texts_to_sequences(raw_docs_val)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))

#pad sequences
word_seq_train = pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = pad_sequences(word_seq_test, maxlen=max_seq_len)
word_seq_val = pad_sequences(word_seq_val, maxlen=max_seq_len)


embed_dim = 100 

#embedding matrix
print('preparing embedding matrix...')
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


##BERT 
bert_tokenizer_transformer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def prepareBertInput(tokenizer,sentences):
    attention_mask=[]
    input_ids=[]
    tokenized = sentences.apply((lambda x: tokenizer.encode(str(x), add_special_tokens=True)))
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    for sentence in sentences:
        tokenized2=tokenizer.encode_plus(str(sentence),  max_length=512, pad_to_max_length=True,add_special_tokens=True)
        attention_mask.append(tokenized2['attention_mask'])
        input_ids.append(tokenized2['input_ids'])

    return input_ids , attention_mask, max_len,tokenized 


train_inputs, train_masks, max_len,tokenized =prepareBertInput(bert_tokenizer_transformer,xBTrain )
valid_inputs, valid_masks, max_len_valid,tokenized_valid =prepareBertInput(bert_tokenizer_transformer,xBValid )
test_inputs, test_masks, max_len_test,tokenized_test =prepareBertInput(bert_tokenizer_transformer,xBTest )


train_inputs=tf.constant(train_inputs)
valid_inputs=tf.constant(valid_inputs)
test_inputs=tf.constant(test_inputs)
train_masks=tf.constant(train_masks)
valid_masks=tf.constant(valid_masks)
test_masks=tf.constant(test_masks)


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

num_filters = 100
filter_sizes=[2,3,4,5]
training_dropout_keep_prob=0.5

model_input = Input(shape=(max_seq_len, ), name="text1")
  
z =  Embedding(nb_words + 1, 
                  embed_dim, 
                   input_length=max_seq_len, embeddings_regularizer=regularizers.l2(0.003),
                   name="embedding")(model_input)

input_ids_in = tf.keras.layers.Input(shape=(512,), name='input_token', dtype='int32')
input_masks_in = tf.keras.layers.Input(shape=(512,), name='masked_token', dtype='int32') 
model_hf2 = TFBertModel.from_pretrained('bert-base-uncased')
embedding_layer = model_hf2([input_ids_in,input_masks_in])[0]

 
    # Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,                         
                     kernel_size=sz,
                     padding="valid",
                     activation="relu",
                     strides=1)(z)
    window_pool_size =  max_seq_len  - sz + 1 
    conv = MaxPooling1D(pool_size=window_pool_size)(conv)  
    conv = Flatten()(conv)
    conv_blocks.append(conv)

#concatenate
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

query_seq_encoding2 = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer)

z1 = layers.concatenate([z, query_seq_encoding2])
z1 = Dropout(training_dropout_keep_prob)(z1)

#score prediction
model_output = Dense(num_classes, activation="sigmoid", name="modeloutput")(z1)
## softmax is for multilabel

 
    #creating model
model = Model(
     inputs = [input_ids_in, input_masks_in, model_input],
     outputs = [model_output],
    )
model.compile(loss="binary_crossentropy", optimizer="adam",  metrics=['acc'])
      
print(model.summary())


history=model.fit(
    [train_inputs,train_masks,word_seq_train],y_train, 
                  batch_size=16, epochs=600, 
                  validation_data =([valid_inputs, valid_masks,word_seq_val],y_valid), 
                  verbose=2)



y_out = model.predict(
    [test_inputs, test_masks, word_seq_test],
    batch_size=4,
)
y_pred = np.where(y_out > 0.5, 1, 0)


print(classification_report(y_test, y_pred,digits=4))


import pydot
dot_img_file='model_cardio_cnntext_bert.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


with open('output_CNNText_BERT_cardio_dis3000_ecg512_T100SG.txt', 'w') as f:
    print('classification report', classification_report(y_test, y_pred,digits=4),file=f)
    print(model.summary(), file=f)
    




