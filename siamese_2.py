
# coding: utf-8

# # Siamese network 2, 03062017

# In[7]:

from datetime import datetime
from IPython.display import SVG

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Lambda
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ProgbarLogger, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from utils import load_embeddings, extract_questions_from_dataframe, save_submission



# In[2]:

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


# In[4]:

questions_A.shape


# In[13]:



def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(
        K.maximum(
            K.sum(K.square(x - y), axis=1, keepdims=True), 
            K.epsilon()
        )
    )

shared_lstm_layer = LSTM(
    units=200,
    return_sequences=False,
    dropout=0.4,
    recurrent_dropout=0.3
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
sentence_representation_A = shared_lstm_layer(embeddings_A)
normalized_A = BatchNormalization()(sentence_representation_A)

input_B = Input(shape=(current_config.MAX_SENTENCE_LENGTH,))
embeddings_B = shared_embedding_layer(input_B)
sentence_representation_B = shared_lstm_layer(embeddings_B)
normalized_B = BatchNormalization()(sentence_representation_B)

distance = Lambda(euclidean_distance)(
    [normalized_A, normalized_B]
)

predictions = Dense(1, activation='sigmoid')(distance)

model = Model(inputs=[input_A, input_B], outputs=predictions)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[14]:

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(
    monitor='val_loss',
    filepath='/output/' + current_config.stamp(comment='1') + '.h5',
    save_best_only=True, 
    save_weights_only=True
)


# In[15]:


training_logs = model.fit(
    x=[questions_A, questions_B], 
    y=labels, 
    epochs=100, 
    batch_size=1024,
    class_weight={0: 1.309028344, 1: 0.472001959},
    validation_split=0.2, 
    callbacks=[early_stopping, model_checkpoint]
)


# In[ ]:

model.save('/output/model_final.bin')

