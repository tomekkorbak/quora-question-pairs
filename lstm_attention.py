
# coding: utf-8

# # Two bidirectional LSTMs with attention, 04062017

# In[11]:

from datetime import datetime

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Lambda, RepeatVector, merge, Permute, Reshape
from keras.layers.merge import concatenate, multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from utils import load_embeddings, extract_questions_from_dataframe, save_submission



# In[3]:

class Config(object):
    VOCABULARY_SIZE = 1193514
    EMBEDDING_DIMENSION = 200
    OFFSET = 3
    OOV_TOKEN = 0  # out of vocabulary
    EOS_TOKEN = 1  # end of sentence
    PAD_TOKEN = 2  # padding to max sentence length
    MAX_SENTENCE_LENGTH = 60
    DENSE_LAYER_SIZE = 150
    DROPOUT = 0.4
    
    def stamp(self, comment):
        return '{date:%Y%m%d_%H%M}_{comment}'.format(
            date=datetime.now(), comment=comment)


# In[4]:

train_dataframe = pd.read_csv('/train/train.csv')
current_config = Config()

embedding_weights, word2idx = load_embeddings(
    '/glove/glove.txt',
    config=current_config
)

questions_A, questions_B, labels = extract_questions_from_dataframe(
    train_dataframe, 
    config=current_config,
    word2idx=word2idx,
    prediction_mode=False
)



# In[14]:

def shared_attention(inputs):
    a = Permute((2, 1))(inputs)
    a = Reshape(
        target_shape=(current_config.EMBEDDING_DIMENSION, 
                      current_config.MAX_SENTENCE_LENGTH)
    )(a)
    a = Dense(current_config.MAX_SENTENCE_LENGTH, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1))(a)  
    a = RepeatVector(n=current_config.EMBEDDING_DIMENSION)(a)
    a_probs = Permute(dims=(2, 1))(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

shared_lstm_layer_1 = Bidirectional(
    LSTM(units=100, 
         return_sequences=True, 
         dropout=0.4)
)

shared_lstm_layer_2 = Bidirectional(
    LSTM(units=100, 
         return_sequences=False, 
         dropout=0.4)
)

shared_embedding_layer = Embedding(
    input_dim=current_config.VOCABULARY_SIZE + current_config.OFFSET, 
    output_dim=current_config.EMBEDDING_DIMENSION, 
    input_length=current_config.MAX_SENTENCE_LENGTH,
    weights=[embedding_weights],
    trainable=False
)

input_A = Input(shape=(current_config.MAX_SENTENCE_LENGTH,))
embeddings_A = shared_embedding_layer(input_A)
initial_representation_A = shared_lstm_layer_1(embeddings_A)
norm_initial_representation_A = BatchNormalization()(initial_representation_A)
attended_representation_A = shared_attention(initial_representation_A)
sentence_representation_A = shared_lstm_layer_2(attended_representation_A)

input_B = Input(shape=(current_config.MAX_SENTENCE_LENGTH,))
embeddings_B = shared_embedding_layer(input_B)
initial_representation_B = shared_lstm_layer_1(embeddings_B)
norm_initial_representation_B = BatchNormalization()(initial_representation_B)
attended_representation_B = shared_attention(initial_representation_B)
sentence_representation_B = shared_lstm_layer_2(attended_representation_B)

merged_model = concatenate([sentence_representation_A, sentence_representation_B])
dropout_1 = Dropout(current_config.DROPOUT)(merged_model)
dense_1 = Dense(current_config.DENSE_LAYER_SIZE)(dropout_1)
dropout_2 = Dropout(current_config.DROPOUT)(dense_1)
merged = BatchNormalization()(dropout_2)

predictions = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[input_A, input_B], outputs=predictions)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[15]:

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
model_checkpoint = ModelCheckpoint(
    monitor='val_loss',
    filepath='/output/' + current_config.stamp(comment='1') + '.h5',
    save_best_only=True, 
    save_weights_only=True
)


# In[16]:

training_logs = model.fit(
    x=[questions_A, questions_B], 
    y=labels, 
    epochs=150, 
    batch_size=1024,
    validation_split=0.2, 
    callbacks=[early_stopping, model_checkpoint]
)


# In[ ]:

model.save('/output/model_final.h5')

